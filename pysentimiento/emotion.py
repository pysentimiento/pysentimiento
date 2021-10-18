import pandas as pd
import os
import pathlib
from glob import glob
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from ..preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""

id2label = {
    0: 'others',
    1: 'joy',
    2: 'sadness',
    3: 'anger',
    4: 'surprise',
    5: 'disgust',
    6: 'fear',
}

label2id = {v:k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent.parent
data_dir = os.path.join(project_dir, "data")
emotion_dir = os.path.join(data_dir, "emoevent")


paths = {
    "es": {
        "train": os.path.join(emotion_dir, "train_es.csv"),
        "test": os.path.join(emotion_dir, "test_es.csv"),
    },
    "en": {
        "train": os.path.join(emotion_dir, "train_en.csv"),
        "test": os.path.join(emotion_dir, "test_en.csv"),
    }
}

def load_df(path):
    """
    Load TASS dataset
    """
    df =  pd.read_csv(path)
    return df


def load_datasets(lang="es", random_state=2021, preprocessing_args={}):
    """
    Load emotion recognition datasets
    """

    train_df = load_df(paths[lang]["train"])
    test_df = load_df(paths[lang]["test"])
    train_df, dev_df = train_test_split(train_df, stratify=train_df["label"], random_state=random_state)


    for df in [train_df, dev_df, test_df]:
        for label, idx in label2id.items():
            df.loc[df["label"] == label, "label"] = idx
        df["label"] = df["label"].astype(int)


    preprocess = lambda x: preprocess_tweet(x, lang=lang, **preprocessing_args)

    train_df.loc[:, "text"] = train_df["text"].apply(preprocess)
    dev_df.loc[:, "text"] = dev_df["text"].apply(preprocess)
    test_df.loc[:, "text"] = test_df["text"].apply(preprocess)

    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=len(id2label), names=[id2label[k] for k in sorted(id2label.keys())])
    })

    train_dataset = Dataset.from_pandas(train_df, features=features)
    dev_dataset = Dataset.from_pandas(dev_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    return train_dataset, dev_dataset, test_dataset
