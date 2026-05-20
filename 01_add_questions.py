import pandas as pd
from pathlib import Path
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

def prepare_true_false_dataset():
    dfs = [] 
    for data_file in (DATA_DIR / "true_false_dataset" / "original_true_false_dataset").glob("*true_false.csv"):
        new = pd.read_csv(data_file)

        new["statement"] = (
            new["statement"].str.replace(
                r'["\'`´""'']', '', regex=True).str.replace(
                    r'([.!?])\1+$', r'\1', regex=True).str.strip()
        )

        new['category'] = data_file.name.replace("_true_false.csv", "")
        
        new = new.sample(frac=1, random_state=42).reset_index(drop=True)
        n_deception = 15
        n_remaining = len(new) - n_deception
        n_train = int(0.6 * n_remaining)
        n_validate = int(0.2 * n_remaining)
        
        splits = (['deception_task'] * n_deception + 
                ['train'] * n_train + 
                ['validate'] * n_validate + 
                ['test'] * (n_remaining - n_train - n_validate))
        
        new['split'] = splits
        dfs.append(new)

    df = pd.concat(dfs, ignore_index=True)
    return df

def add_questions(df):

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    few_shots = [
        {
            "role": "system",
            "content": "You provide questions that a given statement could be an answer to. You do not have to fact-check the statement or mention its potential incorrectness. You should only give one single question."
        },
        {
            "role": "user",
            "content": "There are no specific human uses for skunk."
        },
        {
            "role": "assistant",
            "content": "What are the specific human uses for skunk?"
        },
        {
            "role": "user",
            "content": "The gazelle has distinctive orange and black stripes and is an apex predator."
        },
        {
            "role": "assistant",
            "content": "What are the distinctive features and diet of the gazelle?"
        },
        {
            "role": "user",
            "content": "Johnson & Johnson is a holding company that engages in the research and development, manufacture and sale of products in the healthcare field."
        },
        {
            "role": "assistant",
            "content": "What type of company is Johnson & Johnson and what products do they deal with?"
        },
        {
            "role": "user",
            "content": "Lanthanum appears in its standard state as liquid."
        },
        {
            "role": "assistant",
            "content": "What is the standard state of Lanthanum?"
        },
        {
            "role": "user",
            "content": "The peacock has the atomic number of mammal."
        },
        {
            "role": "assistant",
            "content": "Can you tell me a fact about peacocks?"
        },
        {
            "role": "user",
            "content": "The koala uses swimming for locomotion."
        },
        {
            "role": "assistant",
            "content": "What type of movement does the Koala use for locomotion?"
        },
        {
            "role": "user",
            "content": "The tapir has a diet of carnivore."
        },
        {
            "role": "assistant",
            "content": "What is the diet of a tapir?"
        },
        {
            "role": "user",
            "content": "Cockburn Town is a name of a country."
        },
        {
            "role": "assistant",
            "content": "What is Cockburn Town?"
        },
        {
            "role": "user",
            "content": "Kuala Lumpur is a name of a country."
        },
        {
            "role": "assistant",
            "content": "What is Kuala Lumpur?"
        },
        {
            "role": "user",
            "content": "The human body has 12 pairs of legs."
        },
        {
            "role": "assistant",
            "content": "How many legs does the human body have?"
        },
        {
            "role": "user",
            "content": "The greenhouse effect influences Earth's temperature."
        },
        {
            "role": "assistant",
            "content": "What does the greenhouse effect influence?"
        },
    ]
    for idx, row in df.iterrows():
        messages = few_shots + [
            {
                "role": "user",
                "content": f"{row['statement']}"
            },
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = output[0][input_ids.shape[-1]:]
        question = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        df.loc[idx, 'question'] = question
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(df)} statements.")
            if idx % 100 == 0:
                df.to_csv(DATA_DIR / "true_false_dataset_with_questions.csv", index=False)
    
    return df

def main():
    df = prepare_true_false_dataset()
    df = add_questions(df)
    df.to_csv(DATA_DIR / "true_false_dataset" / "true_false_dataset_with_questions.csv", index=False)

if __name__ == "__main__":
    main()