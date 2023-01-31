"""
Run sentiment experiments
"""
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_tweet
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments

task_name = "irony"

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

label2id = {v: k for k, v in id2label.items()}


def load_datasets(lang, data_path=None, limit=None, random_state=20202021, preprocess=True, preprocess_args={}):
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
            x, lang=lang, **preprocess_args)
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

    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(
            range(min(limit, len(train_dataset))))
        dev_dataset = dev_dataset.select(range(min(limit, len(dev_dataset))))
        test_dataset = test_dataset.select(
            range(min(limit, len(test_dataset))))

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

    load_extra_args = extra_args[base_model] if base_model in extra_args else {
    }

    ds = load_datasets(
        lang=lang, **load_extra_args)

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
    ds = load_datasets(lang=lang, **extra_args.get(model_name, {}))

    def model_init():
        model, _ = load_model(model_name, id2label)
        return model

    _, tokenizer = load_model(model_name, id2label)

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
