import pandas as pd
from transformers import AutoTokenizer
from utils import *

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

input_path = './data/uspatent/'
df = pd.read_csv(input_path + 'train.csv')
df_title = pd.read_csv(input_path + 'titles.csv')

df.merge(df_title, how='left', left_on='context', right_on='code')
df = df[['id'], ['anchor'], ['target'], ['context'], ['score'], ['title']]

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 1
for f, (t_, v_) in enumerate(kf.split(X=df, y=df['anchor'], groups=df['anchor'])):
    df.loc[v_, 'fold'] = f

df['input'] = df['anchor'] + tokenizer.sep_token + df['title'].apply(str.lower)

import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, df):
        self.inputs = df['input'].values.astype(str)
        self.targets = df['target'].values.astype(str)
        self.label = df['score'].values.astype(str)

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, item):
        inputs = self.inputs[item]
        targets = self.targets[item]
        label = self.label[item]

        inputs = tokenizer(inputs,
                           targets,
                           max_length=64,
                           padding='max_length',
                           truncation=True)
        return {**inputs,
                'label': torch.as_tensor(label, dtype=torch.float)}


if __name__ == '__main__':
    # print(tokenizer("hellow, this is one sentence", 'and this is another'))
    # print(df_title)
    pass
