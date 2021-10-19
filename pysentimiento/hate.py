
"""
Run hatEval experiments
"""
import pandas as pd
import os
import pathlib
from datasets import Dataset, Value, ClassLabel, Features
from .preprocessing import preprocess_tweet, extra_args
from .training import load_model, train_model

id2label = {
    0: 'ok',
    1: 'hateful',
}

label2id = {v:k for k, v in id2label.items()}

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data", "hate")


def load_datasets(lang, train_path=None, dev_path=None, test_path=None, limit=None,
    random_state=2021, preprocessing_args={} ):
    """
    Load emotion recognition datasets
    """



    train_path = train_path or os.path.join(data_dir, "hateval2019_es_train.csv")
    dev_path = dev_path or os.path.join(data_dir, "hateval2019_es_dev.csv")
    test_path = test_path or os.path.join(data_dir, "hateval2019_es_test.csv")


    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    preprocess = lambda x: preprocess_tweet(x, lang=lang, **preprocessing_args)

    for df in [train_df, dev_df, test_df]:
        df["label"] = df["HS"].astype(int)
        df["text"] = df["text"].apply(preprocess)


    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=len(id2label), names=[id2label[k] for k in sorted(id2label.keys())]),
        'TR': ClassLabel(num_classes=len(id2label), names=["GROUP", "INDIVIDUAL"]),
    })

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
    base_model, lang, epochs=5, batch_size=32, eval_batch_size=16,
    warmup_ratio=.1, limit=None, accumulation_steps=1,
    **kwargs,
    ):
    """
    Train function
    """

    train_dataset, dev_dataset, test_dataset = load_datasets(
        lang=lang,
        preprocessing_args=extra_args.get(base_model, {})
    )


    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))


    model, tokenizer = load_model(base_model,
        id2label=id2label,
        label2id=label2id
    )


    return train_model(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        epochs=epochs, batch_size=batch_size, class_weight=None,
        warmup_ratio=warmup_ratio, accumulation_steps=accumulation_steps,
    )
