"""
Run sentiment experiments
"""
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_tweet, get_preprocessing_args
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments

task_name = "irony"

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
sentiment_dir = os.path.join(data_dir, "irony")

id2label = {
    0: 'not ironic',
    1: 'ironic',
}

label2id = {v: k for k, v in id2label.items()}


def load_from_file(lang, data_path=None, limit=None, random_state=20202021, preprocess=True, preprocessing_args={}):
    """
    Load sentiment datasets
    """

    features = Features({
        'id': Value('string'),
        'text': Value('string'),
        'topic': Value('string'),
        'lang': Value('string'),
        'label': ClassLabel(num_classes=2)
    })
    data_path = data_path or os.path.join(sentiment_dir, "irosva_dataset.csv")
    df = pd.read_csv(data_path)
    df["label"] = df["is_ironic"]

    if preprocess:

        def preprocess_fn(x): return preprocess_tweet(
            x, lang=lang, **preprocessing_args)
        df["text"] = df["text"].apply(preprocess_fn)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df, dev_df = train_test_split(
        train_df, stratify=train_df["label"], random_state=random_state,
        test_size=0.25,
    )

    train_dataset = Dataset.from_pandas(
        train_df[features.keys()],
        features=features,
        preserve_index=False
    )
    dev_dataset = Dataset.from_pandas(
        dev_df[features.keys()],
        features=features,
        preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[features.keys()],
        features=features,
        preserve_index=False
    )

    return DatasetDict(
        train=train_dataset,
        dev=dev_dataset,
        test=test_dataset
    )


def load_datasets(lang, preprocess=True, preprocessing_args={}):
    """
    Load sentiment datasets
    """

    if lang in {"es", "en", "pt"}:
        ds = load_dataset(f"pysentimiento/{lang}_irony")
    elif lang == "it":
        ds = load_dataset("pysentimiento/it_sentipolc16")
        ds = ds.map(lambda ex: {"label": ex["iro"]})
    else:
        raise ValueError(f"Language {lang} not supported for irony detection")

    if preprocess:

        def preprocess_fn(ex):
            return {"text": preprocess_tweet(ex["text"], lang=lang, **preprocessing_args)}
        ds = ds.map(preprocess_fn, batched=False)

    return ds


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in ["es", "en", "it", "pt"]


def train(
    base_model, lang="es", use_defaults_if_not_tuned=False,
    **kwargs
):
    """
    """

    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(base_model, lang=lang)
    )

    training_args = get_training_arguments(
        base_model, task_name=task_name, lang=lang,
        metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned
    )

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        training_args=training_args, lang=lang, use_defaults_if_not_tuned=use_defaults_if_not_tuned,
        **kwargs
    )


def hp_tune(model_name, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(model_name, lang=lang)
    )

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
