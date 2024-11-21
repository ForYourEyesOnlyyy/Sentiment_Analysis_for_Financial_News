import os
import pandas as pd

load_dir = './data/raw'
input_file_names = [
    'Sentences_AllAgree.txt', 'Sentences_75Agree.txt', 'Sentences_66Agree.txt'
]
save_dir = './data/unprocessed/'
os.makedirs(save_dir, exist_ok=True)

dataframes = []

for file_name in input_file_names:
    input_file_path = os.path.join(load_dir, file_name)
    with open(input_file_path, 'r', encoding='ISO-8859-1') as file:
        data = file.readlines()

    data_tuples = [(line.split('@')[0].strip(), line.split('@')[1].strip())
                   for line in data if '@' in line]

    dataframes.append(pd.DataFrame(data_tuples, columns=['text', 'sentiment']))

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

combined_df.to_csv(os.path.join(save_dir, 'combined_phrasebank.csv'),
                   index=False)
print(
    "Shuffled and combined 'takala/financial_phrasebank ' dataset saved successfully."
)
