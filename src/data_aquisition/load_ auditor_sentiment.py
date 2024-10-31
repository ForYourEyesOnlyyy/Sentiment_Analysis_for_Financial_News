import os
import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset =load_dataset("FinanceInc/auditor_sentiment")

# Combine train and validation datasets
combined_df = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Ensure save directory exists
save_dir = './data/unprocessed/'
os.makedirs(save_dir, exist_ok=True)

# Save the shuffled dataframe to save_dir
combined_df.to_csv(os.path.join(save_dir, 'combined_auditor.csv'), index=False)

print("Shuffled and combined 'FinanceInc/auditor_sentiment' dataset saved successfully.")

