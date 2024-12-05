import pandas as pd


def clean_auditor(df):
    df = df.rename(columns={'sentence': 'text', 'label': 'label'})
    df['label'] = df['label'].replace({1: 2, 2: 1})
    return df


def clean_phrasebank(df):
    labels = {"negative": 0, "positive": 1, "neutral": 2}
    df = df.rename(columns={'text': 'text', 'sentiment': 'label'})
    df['label'] = df['label'].replace(labels)
    return df


def set_column_types(df):
    """
    Sets the 'text' column to string type and 'label' column to integer type.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.

    Returns:
    - pd.DataFrame: The DataFrame with updated column types.
    """
    # Set 'text' column as string
    df['text'] = df['text'].astype(str)

    # Set 'label' column as integer
    df['label'] = df['label'].astype(int)

    return df


# Load the datasets
zero_shot = pd.read_csv('./data/unprocessed/combined_zeroshot.csv')
auditor = pd.read_csv('./data/unprocessed/combined_auditor.csv')
phrasebank = pd.read_csv('./data/unprocessed/combined_phrasebank.csv')

# Process the datasets
auditor = clean_auditor(auditor)
phrasebank = clean_phrasebank(phrasebank)

full_df = pd.concat([zero_shot, auditor, phrasebank], ignore_index=True)
full_df = full_df.sample(frac=1).reset_index(drop=True)

# Set column types
full_df = set_column_types(full_df)

# Save the combined dataset
full_df.to_csv('./data/processed/combined_full.csv', index=False)
print("Combined dataset saved successfully.")
