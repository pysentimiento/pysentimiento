import pandas as pd
import os
import pathlib
from glob import glob
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ..preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""

id2label = [
    'others',
    'joy',
    'sadness',
    'anger',
    'surprise',
    'disgust',
    'fear',
]

label2id = {v:k for k, v in enumerate(id2label)}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent.parent
data_dir = os.path.join(project_dir, "data")
tass_dir = os.path.join(data_dir, "tass2020")
emotion_dir = os.path.join(tass_dir, "task2")


def load_df(path):
    """
    Load TASS dataset
    """
    df =  pd.read_table(path, index_col=0, names=["id", "text", "label"], skiprows=1)
    df["label"] = df["label"].apply(lambda x: x.strip())
    return df


def load_datasets():
    """
    Load task-2 emotion recognition datasets
    """

    train_df = load_df(os.path.join(emotion_dir, "train.tsv"))
    dev_df = load_df(os.path.join(emotion_dir, "dev.tsv"))

    dev_df["label"] = dev_df["label"].apply(lambda x: x.strip())
    train_df["label"].value_counts()



    for df in [train_df, dev_df]:
        for label, idx in label2id.items():
            df.loc[df["label"] == label, "label"] = idx
        df["label"] = df["label"].astype(int)


    train_df["text"] = train_df["text"].apply(preprocess_tweet)
    dev_df["text"] = dev_df["text"].apply(preprocess_tweet)

    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=len(id2label), names=id2label)
    })


    train_dataset = Dataset.from_pandas(train_df, features=features)
    dev_dataset = Dataset.from_pandas(dev_df, features=features)

    return train_dataset, dev_dataset
