import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict
from sklearn.model_selection import train_test_split
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments
from .preprocessing import preprocess_tweet

"""
Lo pongo así por huggingface
"""

task_name = "emotion"

id2label = {
    0: 'others',
    1: 'joy',
    2: 'sadness',
    3: 'anger',
    4: 'surprise',
    5: 'disgust',
    6: 'fear',
}

label2id = {v: k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
emotion_dir = os.path.join(data_dir, "emotion")


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
    df = pd.read_csv(path)
    return df


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in paths


def load_datasets(lang="es", random_state=2021, preprocessing_args={}, preprocess=True):
    """
    Load emotion recognition datasets
    """

    train_df = load_df(paths[lang]["train"])
    test_df = load_df(paths[lang]["test"])
    train_df, dev_df = train_test_split(
        train_df, stratify=train_df["label"], random_state=random_state)

    for df in [train_df, dev_df, test_df]:
        for label, idx in label2id.items():
            df.loc[df["label"] == label, "label"] = idx
        df["label"] = df["label"].astype(int)

    if preprocess:
        def preprocess_fn(x): return preprocess_tweet(
            x, lang=lang, **preprocessing_args)

        train_df.loc[:, "text"] = train_df["text"].apply(preprocess_fn)
        dev_df.loc[:, "text"] = dev_df["text"].apply(preprocess_fn)
        test_df.loc[:, "text"] = test_df["text"].apply(preprocess_fn)

    features = Features({
        'id': Value('string'),
        'text': Value('string'),
        'label': ClassLabel(num_classes=len(id2label), names=[id2label[k] for k in sorted(id2label.keys())])
    })

    train_dataset = Dataset.from_pandas(
        train_df[features.keys()], features=features, preserve_index=False)
    dev_dataset = Dataset.from_pandas(
        dev_df[features.keys()], features=features, preserve_index=False)
    test_dataset = Dataset.from_pandas(
        test_df[features.keys()], features=features, preserve_index=False)

    return DatasetDict(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset
    )


def train(
    base_model, lang="es", use_defaults_if_not_tuned=False,
    **kwargs
):
    """
    """
    ds = load_datasets(lang=lang)

    training_args = get_training_arguments(
        base_model, task_name=task_name, lang=lang,
        metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned
    )

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        training_args=training_args, lang=lang, **kwargs
    )


def hp_tune(model_name, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    ds = load_datasets(lang=lang)

    def model_init():
        model, _ = load_model(model_name, id2label, lang=lang)
        return model

    _, tokenizer = load_model(model_name, id2label, lang=lang)

    config_info = {
        "model": model_name,
        "task": task_name,
        "lang": lang,
    }

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{lang}-{model_name}",
        group_name=f"swp-{task_name}-{lang}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
        **kwargs,
    )
