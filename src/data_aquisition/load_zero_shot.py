import os
import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Combine train and validation datasets
combined_df = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Ensure save directory exists
save_dir = './data/unprocessed/'
os.makedirs(save_dir, exist_ok=True)

# Save the shuffled dataframe to save_dir
combined_df.to_csv(os.path.join(save_dir, 'combined_zeroshot.csv'), index=False)

print("Shuffled and combined 'zeroshot/twitter-financial-news-sentiment' dataset saved successfully.")

