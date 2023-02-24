import pandas as pd
import os
import pathlib
from glob import glob
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict
from .preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""
id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
label2id = {v: k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
tass_dir = os.path.join(data_dir, "sentiment")


def get_lang(file):
    """
    Get language of TASS dataset
    """
    return os.path.splitext(os.path.basename(file))[0]


def load_df(path, test=False):
    """
    Load TASS dataset
    """
    dialect = get_lang(path)

    if test:
        df = pd.read_table(path, names=["id", "text"], index_col=0)
        label_path = os.path.join(
            os.path.dirname(path),
            "labels",
            f"{dialect.lower()}.tsv"
        )

        labels = pd.read_table(label_path, names=["id", "label"], index_col=0)

        df["polarity"] = labels["label"]
    else:
        df = pd.read_table(path, names=["id", "text", "polarity"], index_col=0)

    df["lang"] = dialect

    for label, idx in label2id.items():
        polarity_label = {"NEG": "N", "NEU": "NEU", "POS": "P"}[label]
        df.loc[df["polarity"] == polarity_label, "label"] = idx

    return df


def load_datasets(preprocessing_args={}, preprocess=True, return_df=False, **kwargs):
    """
    Return train, dev, test datasets
    """

    df = pd.read_csv(os.path.join(tass_dir, "tass.csv"))

    df["label"] = df["polarity"].apply(lambda x: label2id[x])

    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(len(train_df), len(dev_df), len(test_df))

    """
    Tokenize tweets
    """

    if preprocess:
        def preprocess_with_args(x): return preprocess_tweet(
            x, **preprocessing_args)

        train_df["text"] = train_df["text"].apply(preprocess_with_args)
        dev_df["text"] = dev_df["text"].apply(preprocess_with_args)
        test_df["text"] = test_df["text"].apply(preprocess_with_args)

    if return_df:
        return train_df, dev_df, test_df

    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=3, names=["neg", "neu", "pos"])
    })

    columns = ["text", "lang", "label"]

    train_dataset = Dataset.from_pandas(
        train_df[columns],
        features=features,
        preserve_index=False
    )
    dev_dataset = Dataset.from_pandas(
        dev_df[columns],
        features=features,
        preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[columns],
        features=features,
        preserve_index=False
    )

    return DatasetDict(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset
    )
