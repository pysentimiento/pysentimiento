"""
Run sentiment experiments
"""
import torch
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_tweet
from .training import train_model

extra_args = {
    "vinai/bertweet-base": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
sentiment_dir = os.path.join(data_dir, "irony")

id2label = {
    0: 'not ironic',
    1: 'ironic',
}

label2id = {v:k for k, v in id2label.items()}


def load_datasets(lang, data_path=None, limit=None, random_state=20202021, preprocess=True, preprocess_args={}):
    """
    Load sentiment datasets
    """
    features = Features({
        'text': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=2)
    })
    data_path = data_path or os.path.join(sentiment_dir, "irosva_dataset.csv")
    df = pd.read_csv(data_path)
    df["label"] = df["is_ironic"]


    if preprocess:

        preprocess_fn = lambda x: preprocess_tweet(x, lang=lang, **preprocess_args)
        df["text"] = df["text"].apply(preprocess_fn)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df, dev_df = train_test_split(
        train_df, stratify=train_df["label"], random_state=random_state,
        test_size=0.25,
    )

    train_dataset = Dataset.from_pandas(train_df, features=features)
    dev_dataset = Dataset.from_pandas(dev_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)


    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))
        dev_dataset = dev_dataset.select(range(min(limit, len(dev_dataset))))
        test_dataset = test_dataset.select(range(min(limit, len(test_dataset))))


    return train_dataset, dev_dataset, test_dataset


def train(
    base_model, lang="es", epochs=5, batch_size=32,
    limit=None, **kwargs
):
    """
    """

    load_extra_args = extra_args[base_model] if base_model in extra_args else {}

    train_dataset, dev_dataset, test_dataset = load_datasets(lang=lang, **load_extra_args)

    kwargs = {
        **kwargs,
        **{
            "id2label" : id2label,
            "epochs": epochs,
            "batch_size": batch_size,
            "limit": limit,
            "lang": lang,
        }
    }

    return train_model(base_model, train_dataset, dev_dataset, test_dataset, **kwargs)
