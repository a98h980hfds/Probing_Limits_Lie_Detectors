import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

def main():
    data_file = DATA_DIR / "true_false_dataset" / "true_false_dataset_with_questions.csv"
    df = pd.read_csv(data_file)
    df_deception_task = df[df['split'] == 'deception_task']
    df_deception_task = df_deception_task.reset_index(drop=True)
    df_responses = pd.read_csv(DATA_DIR / "true_false_dataset" / "honest_lie_deception_responses.csv").reset_index(drop=True)
    df_deception_task[['honest_response', 'lie_response', 'deception_no_lie_response']] = df_responses[['honest_response', 'lie_response', 'deception_no_lie_response']]
    df_deception_task.to_csv(DATA_DIR / "true_false_dataset" / "deception_task.csv", index=False)

if __name__ == "__main__":
    main()