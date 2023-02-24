"""
NER for LinCE dataset
"""
import numpy as np
from .ner import tokenize_and_align_labels, preprocess_token
from datasets import load_dataset, load_metric, DatasetDict
from seqeval.metrics import f1_score
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from ..training import train_and_eval

metric = load_metric("seqeval")

"""
TODO: this is a hack to make it work with the current version of seqeval
As you can see, I add "B-" to each label because it only works with BIO tags
"""

id2label = [
    'B-VERB',
    'B-PUNCT',
    'B-PRON',
    'B-NOUN',
    'B-DET',
    'B-ADV',
    'B-ADP',
    'B-INTJ',
    'B-CONJ',
    'B-ADJ',
    'B-AUX',
    'B-SCONJ',
    'B-PART',
    'B-PROPN',
    'B-NUM',
    'B-UNK',
    'B-X',
]

label2id = {v: k for k, v in enumerate(id2label)}


def load_datasets(lang="es", preprocess=True):
    """
    Load NER datasets
    """

    lince = load_dataset("lince", "pos_spaeng")

    # This hack is because of seqeval only working with BIO tags

    lince = lince.map(
        lambda x: {"pos": ["B-"+x for x in x["pos"]]}
    )
    """
    TODO: None is for test labels which are not available
    """
    lince = lince.map(
        lambda x: {"labels": [label2id.get(x, None) for x in x["pos"]]}
    )

    if preprocess:
        lince = lince.map(
            lambda x: {
                "words": [preprocess_token(word, lang) for word in x["words"]]
            }
        )

    return DatasetDict(
        train=lince["train"],
        dev=lince["validation"],
        test=lince["test"],
    )


metric = load_metric("seqeval")


def compute_metrics(eval_preds):
    """
    Compute metrics for POS
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(
        predictions=true_predictions, references=true_labels)
    ret = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "macro_f1": f1_score(true_labels, true_predictions, average="macro"),
        "micro_f1": f1_score(true_labels, true_predictions, average="micro"),
        "accuracy": all_metrics["overall_accuracy"],
    }

    for k, v in all_metrics.items():
        if not k.startswith("overall"):
            ret[k + "_f1"] = v["f1"]
            ret[k + "_precision"] = v["precision"]
            ret[k + "_recall"] = v["recall"]

    return ret


def train(
        base_model, lang, epochs=5,
        metric_for_best_model="accuracy",
        **kwargs):

    ds = load_datasets(
        lang=lang
    )

    return train_and_eval(
        base_model,
        train_dataset=ds["train"], dev_dataset=ds["dev"], test_dataset=ds["dev"],
        id2label=id2label, lang=lang, epochs=epochs,
        # Custom stuff for this thing to work
        tokenize_fun=tokenize_and_align_labels,
        auto_class=AutoModelForTokenClassification,
        data_collator_class=DataCollatorForTokenClassification,
        metrics_fun=compute_metrics,
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )
