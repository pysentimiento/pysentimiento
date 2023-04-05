import random
from .config import config
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from .preprocessing import get_preprocessing_args
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments
import logging


logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


task_name = "targeted_sentiment"


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in ["es"]


def load_datasets(lang, preprocess=True, preprocessing_args={}, randomize=True):
    """"
    Load targeted sentiment datasets

    Args:
        lang (str): Language of the dataset
        preprocess (bool, optional): Whether to preprocess the dataset. Defaults to True.
        preprocessing_args (dict, optional): Arguments for the preprocessing. Defaults to {}.
        randomize: Whether to randomize the dataset. Defaults to True.
    Returns:


    """
    if lang != "es":
        raise ValueError(
            f"Language {lang} not supported for Targeted Sentiment Analysis detection")

    ds = load_dataset("pysentimiento/spanish-targeted-sentiment-headlines")

    if not randomize:
        return ds
    else:

        dataset = concatenate_datasets(list(ds.values()))
        test_size = random.normalvariate(0.2, 0.02)

        df = dataset.to_pandas()
        id_noticias = df.groupby("id_noticia").count()["label"]

        train_ids, test_ids = train_test_split(
            id_noticias, test_size=test_size)
        train_ids, dev_ids = train_test_split(train_ids, test_size=test_size)

        df_train = df[df["id_noticia"].isin(train_ids.index)]
        df_dev = df[df["id_noticia"].isin(dev_ids.index)]
        df_test = df[df["id_noticia"].isin(test_ids.index)]

        columns = ["titulo", "id_noticia", "target", "label"]

        features = dataset.features

        train_dataset = Dataset.from_pandas(
            df_train[columns], features=features, preserve_index=False)
        dev_dataset = Dataset.from_pandas(
            df_dev[columns], features=features, preserve_index=False)
        test_dataset = Dataset.from_pandas(
            df_test[columns], features=features, preserve_index=False)

        return DatasetDict(
            train=train_dataset,
            dev=dev_dataset,
            test=test_dataset,
        )


def get_wandb_run_info(base_model, task, lang, untargeted=False, **kwargs):
    # Check if task module has a get_wandb_run_info method

    if untargeted:
        model_name = base_model + "_untargeted"
    else:
        model_name = base_model

    return {
        "project": config["WANDB"]["PROJECT"],
        # Group by model name
        "group": f"{task}-{lang}",
        "job_type": f"{task}-{lang}-{model_name.split('/')[-1]}",
        # Name run by model name
        "config": {
            "model": model_name,
            "task": task,
            "lang": lang,
        }
    }


def hp_tune(model_name, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(model_name, lang=lang),
        # Just not randomize for now
        randomize=False
    )

    logger.info(ds)

    id2label = {k: v for k, v in enumerate(
        ds["train"].features["label"].names)}

    def model_init():
        model, _ = load_model(model_name, id2label, lang=lang)
        return model

    _, tokenizer = load_model(model_name, id2label, lang=lang)

    config_info = {
        "model": model_name,
        "task": task_name,
        "lang": lang,
    }

    def tokenize_fun(batch):
        return tokenizer(
            batch['titulo'], batch["target"], padding=False, truncation=True)

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{lang}-{model_name}",
        group_name=f"swp-{task_name}-{lang}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
        tokenize_fun=tokenize_fun,
        **kwargs,
    )


def train(
    base_model, lang="es", use_defaults_if_not_tuned=False, randomize=True,
    benchmark=True,
    untargeted=False, **kwargs
):

    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(base_model, lang=lang),
        # Just not randomize for now
        randomize=randomize
    )

    logger.info(ds)

    id2label = {k: v for k, v in enumerate(
        ds["train"].features["label"].names)}

    training_args = get_training_arguments(
        base_model, task_name=task_name, lang=lang,
        metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned
    )

    _, tokenizer = load_model(base_model, id2label, lang=lang)

    def tokenize_fun(batch):
        if not untargeted:
            return tokenizer(
                batch['titulo'], batch["target"], padding=False, truncation=True)
        else:
            return tokenizer(
                batch['titulo'], padding=False, truncation=True)

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        tokenize_fun=tokenize_fun,
        training_args=training_args, lang=lang, **kwargs
    )
