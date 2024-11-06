
import numpy as np
import os
import pandas as pd

def analyze_class_distribution(df):
    if 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")
    
    distribution = df['target'].value_counts(normalize=True)
    print("\nClass Distribution:")
    print(distribution)
    print("\nTotal samples:", len(df))
    print("Positive samples:", len(df[df['target'] == 1]))
    print("Negative samples:", len(df[df['target'] == 0]))
    return distribution




train_csv = pd.read_csv('train.csv')
train_csv['filename'] = train_csv['image_name'].apply(lambda x: f"{x}.dcm")
train_csv['target'] = train_csv['target'].astype(np.float32)
analyze_class_distribution(train_csv)
