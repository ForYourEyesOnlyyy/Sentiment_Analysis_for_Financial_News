import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Create the directory for saving samples if it doesn't exist
sample_dir = './data/processed/twitter-financial-news-sentiment/samples'
os.makedirs(sample_dir, exist_ok=True)

# Define the number of samples and rows per sample
num_samples = 3
rows_per_sample = 6000

# Define stratification percentage for each class
# For example: 0.5 means 50% of the samples will be stratified from each class in the dataset
stratify_percentage = 0.9  # Adjust this as needed

# Sampling and saving the stratified data
for i in range(1, num_samples + 1):
    # Convert the Hugging Face Dataset object to a pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Extract the 'label' column for stratification and calculate class counts
    class_counts = df['label'].value_counts()
    
    # Calculate how many samples to take from each class based on stratify_percentage
    rows_per_class = (class_counts * stratify_percentage).astype(int)
    
    # Ensure the total number of rows doesn't exceed the desired sample size
    rows_per_class = rows_per_class.apply(lambda x: min(x, rows_per_sample // len(class_counts)))

    # Stratified sampling: Collect samples from each class
    stratified_sample = pd.concat([
        df[df['label'] == label].sample(n=rows_per_class[label], random_state=i, replace=False)
        for label in class_counts.index
    ])
    
    # Shuffle the final sample and ensure it has the correct total number of rows
    stratified_sample = stratified_sample.sample(frac=1, random_state=i).reset_index(drop=True)
    
    # Save the stratified sample to CSV
    stratified_sample.to_csv(f'{sample_dir}/sample{i}.csv', index=False)
    print(f'Saved stratified sample {i} with {len(stratified_sample)} rows to sample{i}.csv')