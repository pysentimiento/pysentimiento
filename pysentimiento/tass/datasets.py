import pandas as pd
import os
from glob import glob
from datasets import Dataset, Value, ClassLabel, Features
from ..preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""
id2label = {0: 'N', 1: 'NEU', 2: 'P'}
label2id = {v:k for k,v in id2label.items()}


def get_lang(file):
    """
    Get language of TASS dataset
    """
    return os.path.splitext(os.path.basename(file))[0]

def load_df(path):
    """
    Load TASS dataset
    """
    dialect = get_lang(path)

    df = pd.read_table(path, names=["id", "text", "polarity"], index_col=0)
    df["lang"] = dialect

    for label, idx in label2id.items():
        df.loc[df["polarity"] == label, "label"] = idx
    return df

def load_datasets():
    """
    Return train, dev, test datasets
    """
    train_files = glob("data/tass2020/train/*.tsv")
    dev_files = glob("data/tass2020/dev/*.tsv")
    test_files = glob("data/tass2020/test1.1/*.tsv")

    train_dfs = {get_lang(file):load_df(file) for file in train_files}
    dev_dfs = {get_lang(file):load_df(file) for file in dev_files}
    test_dfs = {get_lang(file):load_df(file) for file in test_files}

    train_df = pd.concat(train_dfs.values())
    dev_df = pd.concat(dev_dfs.values())
    test_df = pd.concat(test_dfs.values())

    print(len(train_df), len(dev_df), len(test_df))

    """
    Tokenize tweets
    """

    train_df["text"] = train_df["text"].apply(preprocess_tweet)
    dev_df["text"] = dev_df["text"].apply(preprocess_tweet)
    test_df["text"] = test_df["text"].apply(preprocess_tweet)

    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=3, names=["neg", "neu", "pos"])
    })

    columns = ["text", "lang", "label"]

    train_dataset = Dataset.from_pandas(train_df[columns], features=features)
    dev_dataset = Dataset.from_pandas(dev_df[columns], features=features)
    test_dataset = Dataset.from_pandas(test_df[columns], features=features)

    return train_dataset, dev_dataset, test_dataset