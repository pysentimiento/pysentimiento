from numpy import int64
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""
id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
label2id = {v:k for k,v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent.parent
data_dir = os.path.join(project_dir, "data")
semeval_dir = os.path.join(data_dir, "SemEval2017")


def load_df(path):

    df = pd.read_csv(path, encoding="latin-1")

    matches = {"NEG": "negative", "NEU": "neutral", "POS": "positive"}

    for label, idx in label2id.items():
        replacement = matches[label]
        df.loc[df["label"] == replacement, "label"] = idx

    df["label"] = df["label"].astype(int)
    return df

def load_datasets(seed=2021, preprocessing_args={}):
    """
    Return train, dev, test datasets
    """
    train_df = load_df(
        os.path.join(semeval_dir, "train.csv")
    )

    test_df = load_df(
        os.path.join(semeval_dir, "test.csv")
    )

    train_df, dev_df = train_test_split(train_df, test_size=0.2)

    print(len(train_df), len(dev_df), len(test_df))

    """
    Tokenize tweets
    """

    en_preprocess = lambda x: preprocess_tweet(x, lang="en", **preprocessing_args)

    train_df["text"] = train_df["text"].apply(en_preprocess)
    dev_df["text"] = dev_df["text"].apply(en_preprocess)
    test_df["text"] = test_df["text"].apply(en_preprocess)

    features = Features({
        'id': Value('int64'),
        'text': Value('string'),
        'label': ClassLabel(num_classes=3, names=["NEG", "NEU", "POS"])
    })


    columns = ["text", "id", "label"]

    train_dataset = Dataset.from_pandas(train_df[columns], features=features)
    dev_dataset = Dataset.from_pandas(dev_df[columns], features=features)
    test_dataset = Dataset.from_pandas(test_df[columns], features=features)

    return train_dataset, dev_dataset, test_dataset