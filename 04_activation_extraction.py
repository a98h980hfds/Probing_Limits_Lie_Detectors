import pandas as pd
import h5py
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
import gc

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

MODELS = {
    "mistral-7b-v03": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "num_layers": 42,
        "num_heads": 16,
        "head_dim": 256,
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
    },
}

BATCH_SIZE = 10

def tokenize_batch(message_batch, tokenizer, apply_chat_template=True):
    if apply_chat_template:
        encoded = tokenizer.apply_chat_template(
            message_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    else:
        encoded = tokenizer(
            message_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )["input_ids"]
    
    attention_mask = (encoded != tokenizer.pad_token_id).long()
    encoded = BatchEncoding({
        "input_ids": encoded,
        "attention_mask": attention_mask,
    })
    return encoded

def extract_batch_attention_outputs(message_batch, model, tokenizer, model_key, apply_chat_template=True):
    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]

    encoded = tokenize_batch(message_batch, tokenizer, apply_chat_template)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size = input_ids.shape[0]

    layers = model.model.layers
         
    collected = [None] * num_layers

    def extract_from_tensor(tensor, layer_idx):
        B, S, H = tensor.shape
        reshaped = tensor.view(B, S, num_heads, head_dim)

        final_idx = attention_mask.sum(dim=1) - 1
        out = np.stack([
            reshaped[b, final_idx[b]].detach().cpu().float().numpy()
            for b in range(B)
        ])
        collected[layer_idx] = out

    def make_hook(layer_idx):
        def hook(mod, inputs, outputs):
            extract_from_tensor(inputs[0], layer_idx)
        return hook

    hooks = []
    for i in range(num_layers):
        attn = layers[i].self_attn
        module = attn.o_proj
        hooks.append(module.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(**encoded)

    for h in hooks:
        h.remove()

    attention_tensor = np.stack(collected, axis=1)

    return attention_tensor

def save_activation_batch(output_path, batch_metadata, attention_tensor):
    B = attention_tensor.shape[0]

    with h5py.File(output_path, "a") as f:

        for key, v in batch_metadata.items():
            v = np.asarray(v)

            if v.dtype.kind in ('U', 'O'):
                dt = h5py.string_dtype(encoding='utf-8')
            else:
                dt = v.dtype

            if key in f:
                dset = f[key]
                old_size = dset.shape[0]
                dset.resize(old_size + B, axis=0)
                dset[old_size:] = v
            else:
                dset = f.create_dataset(
                    key,
                    shape=(B,),
                    maxshape=(None,),
                    dtype=dt
                )
                dset[:] = v

        if "attention" in f:
            dset = f["attention"]
            old_size = dset.shape[0]
            dset.resize(old_size + B, axis=0)
            dset[old_size:] = attention_tensor
        else:
            att_shape = attention_tensor.shape
            max_shape = (None,) + att_shape[1:]

            dset = f.create_dataset(
                "attention",
                shape=att_shape,
                maxshape=max_shape,
                chunks=True,
                dtype=attention_tensor.dtype
            )
            dset[:] = attention_tensor

def main():
    df = pd.read_csv(DATA_DIR / "true_false_dataset" / "true_false_dataset_with_questions.csv")

    modes = [
        {
            "name": "with_user_question",
            "apply_chat_template": True,
        },
        {
            "name": "without_user_question",
            "apply_chat_template": False,
        }
    ]

    for model_key in MODELS.keys():
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_key]["model_id"],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type="bfloat16",
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            dtype="bfloat16",
        )
    
        tokenizer = AutoTokenizer.from_pretrained(
            MODELS[model_key]["model_id"],
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        for mode in modes:
            mode_name = mode["name"]
            apply_chat_template = mode["apply_chat_template"]
            
            output_path = DATA_DIR / "attention_outputs" / f"{model_key}_attention_outputs_{mode_name}.h5"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            for idx in range(0, len(df), BATCH_SIZE):
                print(f"Processing index {idx} to {min(idx+BATCH_SIZE, len(df))} of {len(df)} for model {model_key} in mode {mode_name}.")

                batch = df.iloc[idx:idx+BATCH_SIZE]
                
                if apply_chat_template:
                    message_batch = [
                        [
                            {
                                "role": "user",
                                "content": row["question"]
                            },
                            {
                                "role": "assistant",
                                "content": row["statement"]
                            }
                        ]
                        for idx, row in batch.iterrows()
                    ]
                    
                    batch_metadata = {
                        "target_label": [row["label"] for idx, row in batch.iterrows()],
                        "question": [row["question"] for idx, row in batch.iterrows()],
                        "statement": [row["statement"] for idx, row in batch.iterrows()],
                        "category": [row["category"] for idx, row in batch.iterrows()],
                        "split": [row["split"] for idx, row in batch.iterrows()],
                    }
                else:
                    message_batch = [row["statement"] for idx, row in batch.iterrows()]
                    
                    batch_metadata = {
                        "target_label": [row["label"] for idx, row in batch.iterrows()],
                        "statement": [row["statement"] for idx, row in batch.iterrows()],
                        "category": [row["category"] for idx, row in batch.iterrows()],
                        "split": [row["split"] for idx, row in batch.iterrows()],
                    }
                
                attention_tensor = extract_batch_attention_outputs(
                    message_batch, model, tokenizer, model_key, apply_chat_template
                )
                
                save_activation_batch(output_path, batch_metadata, attention_tensor)

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()