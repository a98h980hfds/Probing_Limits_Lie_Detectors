import pandas as pd
import numpy as np
import torch
import pickle
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


def tokenize_single(message, tokenizer, apply_chat_template=True):
    """Tokenize a single message."""
    if apply_chat_template:
        encoded = tokenizer.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
    else:
        # Tokenize raw text without chat template
        encoded = tokenizer(
            message,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )["input_ids"]
    
    attention_mask = torch.ones_like(encoded)
    encoded = BatchEncoding({
        "input_ids": encoded,
        "attention_mask": attention_mask,
    })
    return encoded


def extract_attention_output(message, model, tokenizer, model_key, apply_chat_template=True):
    """Extract attention outputs for a single message."""
    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]

    encoded = tokenize_single(message, tokenizer, apply_chat_template)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    layers = model.model.layers
    
    # collected[l] = (H, D)
    collected = [None] * num_layers

    def extract_from_tensor(tensor, layer_idx):
        B, S, H = tensor.shape
        reshaped = tensor.view(B, S, num_heads, head_dim)
        
        # Pick last token
        final_idx = attention_mask.sum(dim=1) - 1
        out = reshaped[0, final_idx[0]].detach().cpu().float().numpy()
        collected[layer_idx] = out  # (H, D)

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

    # Stack into (L, H, D)
    attention_tensor = np.stack(collected, axis=0)

    return attention_tensor


def classify_with_probe(attention_tensor, probe_data, model_key):
    """Classify using the trained probe."""
    clf = probe_data["clf"]
    scaler = probe_data["scaler"]
    slice_indices = probe_data["slice"]
    
    # Flatten attention tensor: (L, H, D) -> (L * H * D)
    num_layers = MODELS[model_key]["num_layers"]
    num_heads = MODELS[model_key]["num_heads"]
    head_dim = MODELS[model_key]["head_dim"]
    
    flattened = attention_tensor.reshape(num_layers * num_heads * head_dim)
    
    # Select relevant features
    X = flattened[slice_indices].reshape(1, -1)
    
    # Classify
    X_scaled = scaler.transform(X)
    prob = clf.predict_proba(X_scaled)[0, 1]  # Probability of class 1
    pred = int(prob >= 0.5)
    
    return pred, prob


def main():
    # Load the deception task dataset
    deception_df = pd.read_csv(DATA_DIR / "true_false_dataset" / "deception_task.csv")
    
    # Define modes
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
        print(f"processing model {model_key}")
        
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
            dtype=torch.bfloat16,
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
            
            probe_path = DATA_DIR / "trained_probes" / f"{model_key}_{mode_name}_final_probe.pkl"
            with open(probe_path, "rb") as f:
                probe_data = pickle.load(f)
            
            results = []
            
            for idx, row in deception_df.iterrows():
                question = row["question"]
                print(f"\nProcessing question {idx+1}/{len(deception_df)}: {question[:50]}...")
                
                response_types = {
                    "honest": row["honest_response"],
                    "lie": row["lie_response"],
                    "deception_no_lie": row["deception_no_lie_response"]
                }
                
                for response_type, response_text in response_types.items():
                    
                    if apply_chat_template:
                        message = [
                            {
                                "role": "user",
                                "content": question
                            },
                            {
                                "role": "assistant",
                                "content": response_text
                            }
                        ]
                    else:
                        message = response_text
                    
                    attention_tensor = extract_attention_output(
                        message, model, tokenizer, model_key, apply_chat_template
                    )
                    
                    prediction, probability = classify_with_probe(
                        attention_tensor, probe_data, model_key
                    )
                    
                    results.append({
                        "question": question,
                        "response_type": response_type,
                        "response_text": response_text,
                        "category": row["category"],
                        "prediction": prediction,
                        "probability": probability
                    })
            
            results_df = pd.DataFrame(results)
            output_path = DATA_DIR / "probe_classifications" / f"{model_key}_{mode_name}_probe_classifications.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
        
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()