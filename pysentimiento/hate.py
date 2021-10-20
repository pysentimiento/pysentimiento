
"""
Run hatEval experiments
"""
import pandas as pd
import os
import pathlib
import torch
import numpy as np
import logging
from datasets import Dataset, Value, ClassLabel, Features
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from .preprocessing import preprocess_tweet, extra_args
from .training import load_model, train_model



logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data", "hate")



def load_datasets(lang,
    train_path=None, dev_path=None, test_path=None, limit=None,
    random_state=2021, preprocessing_args={} ):
    """
    Load emotion recognition datasets
    """

    train_path = train_path or os.path.join(data_dir, f"hateval2019_{lang}_train.csv")
    dev_path = dev_path or os.path.join(data_dir, f"hateval2019_{lang}_dev.csv")
    test_path = test_path or os.path.join(data_dir, f"hateval2019_{lang}_test.csv")

    logger.info(f"Train path = {train_path}")
    logger.info(f"Dev path = {dev_path}")
    logger.info(f"Test path = {test_path}")


    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    preprocess = lambda x: preprocess_tweet(x, lang=lang, **preprocessing_args)

    for df in [train_df, dev_df, test_df]:
        df["text"] = df["text"].apply(preprocess)


    features = Features({
        'text': Value('string'),
        'HS': ClassLabel(num_classes=2, names=["OK", "HATEFUL"]),
        'TR': ClassLabel(num_classes=2, names=["GROUP", "INDIVIDUAL"]),
        "AG": ClassLabel(num_classes=2, names=["NOT AGGRESSIVE", "AGGRESSIVE"])
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

def get_task_b_metrics(predictions):
    ret = {}

    f1s = []
    precs = []
    recalls = []


    outputs = predictions.predictions
    labels = predictions.label_ids

    for i, cat in enumerate(["HS", "TR", "AG"]):
        cat_labels, cat_preds = labels[:, i], outputs[:, i]

        cat_preds = cat_preds > 0

        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary', zero_division=0,
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower()+"_f1"] = f1
        ret[cat.lower()+"_precision"] = precision
        ret[cat.lower()+"_recall"] = recall



    neg_hs_f1_score = f1_score(1-(outputs[:, 0] > 0), 1 - labels[:, 0])

    ret["macro_hs_f1_score"] = (f1s[0] + neg_hs_f1_score) / 2
    #
    # We calculate EMR in a gated way
    # Block TR and AG if HS is False
    #
    emr_preds = outputs > 0
    ret["emr_no_gating"] = accuracy_score(labels, emr_preds)
    emr_preds[:, 1] = emr_preds[:, 0] & emr_preds[:, 1]
    emr_preds[:, 2] = emr_preds[:, 0] & emr_preds[:, 2]

    ret["emr"] = accuracy_score(labels, emr_preds)

    ret["macro_f1"] = torch.Tensor(f1s).mean()
    ret["macro_precision"] = torch.Tensor(precs).mean()
    ret["macro_recall"] = torch.Tensor(recalls).mean()


    return ret



def train(
    base_model, lang, epochs=5, batch_size=32, eval_batch_size=16,
    warmup_ratio=.1, limit=None, accumulation_steps=1, task_b=False, class_weight=False,
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

    if task_b:
        metrics_fun = get_task_b_metrics
        id2label = {
            0: "hateful",
            1: "targeted",
            2: "aggressive",
        }
    else:
        metrics_fun = None
        id2label = {
            0: 'ok',
            1: 'hateful',
        }


    label2id = {v:k for k, v in id2label.items()}

    model, tokenizer = load_model(base_model,
        id2label=id2label,
        label2id=label2id
    )

    model.config.problem_type = "multi_label_classification" if task_b else "single_label_classification"


    def format_dataset(dataset):
        def get_labels(examples):
            labels = ["HS", "TR", "AG"] if task_b else ["HS"]
            return {'labels': torch.Tensor([examples[k] for k in labels])}
        dataset = dataset.map(get_labels)

        return dataset

    if class_weight:
        class_weight = torch.Tensor([train_dataset[k] for k in ["HS", "TR", "AG"]])
        class_weight = 1 / (2* class_weight.mean(1))


    return train_model(
        model, tokenizer,
        train_dataset, dev_dataset, test_dataset, id2label, format_dataset=format_dataset,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
        warmup_ratio=warmup_ratio, accumulation_steps=accumulation_steps,
        metrics_fun=metrics_fun,
    )
