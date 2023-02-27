
"""
Run hatEval experiments
"""
import pandas as pd
import os
import pathlib
import torch
import numpy as np
import logging
from datasets import Dataset, Value, ClassLabel, Features, DatasetDict, load_dataset
from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from .preprocessing import preprocess_tweet, get_preprocessing_args
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments


logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)

task_name = "hate_speech"

project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data", "hate")

labels_order = ["HS", "TR", "AG"]

# Labels for the Portuguese dataset
pt_labels = ['Sexism', 'Body', 'Racism', 'Ideology', 'Homophobia']


def load_datasets(lang,
                  train_path=None, dev_path=None, test_path=None, limit=None,
                  preprocess=True, preprocessing_args={}):
    """
    Load hate speech datasets

    """
    if lang == "it":
        ds = load_dataset("pysentimiento/it_haspeede")
        ds = ds.map(lambda x: {"labels": torch.Tensor(
            [x["hs"], x["stereotype"]])}, batched=False)
    elif lang == "pt":
        ds = load_dataset("pysentimiento/pt_hate_speech")
        ds = ds.map(lambda x: {"labels": torch.Tensor(
            [x[l] for l in pt_labels])}, batched=False)
    else:
        train_path = train_path or os.path.join(
            data_dir, f"hateval2019_{lang}_train.csv")
        dev_path = dev_path or os.path.join(
            data_dir, f"hateval2019_{lang}_dev.csv")
        test_path = test_path or os.path.join(
            data_dir, f"hateval2019_{lang}_test.csv")

        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        features = Features({
            'id': Value('int64'),
            'text': Value('string'),
            'HS': ClassLabel(num_classes=2, names=["OK", "HATEFUL"]),
            'TR': ClassLabel(num_classes=2, names=["GROUP", "INDIVIDUAL"]),
            "AG": ClassLabel(num_classes=2, names=["NOT AGGRESSIVE", "AGGRESSIVE"])
        })

        train_dataset = Dataset.from_pandas(
            train_df, features=features, preserve_index=False)
        dev_dataset = Dataset.from_pandas(
            dev_df, features=features, preserve_index=False)
        test_dataset = Dataset.from_pandas(
            test_df, features=features, preserve_index=False)

        ds = DatasetDict(
            train=train_dataset,
            dev=dev_dataset,
            test=test_dataset
        )

        ds = ds.map(lambda x: {
            "labels": torch.Tensor([x["HS"], x["TR"], x["AG"]])
        }, batched=False)

    if preprocess:
        def preprocess_fn(x):
            return {
                "text": preprocess_tweet(x["text"], lang=lang, **preprocessing_args)
            }

        ds = ds.map(preprocess_fn, batched=False)
    return ds


def _get_b_metrics(preds, labels):
    ret = {}

    f1s = []
    precs = []
    recalls = []
    original_preds = preds.copy()

    preds[:, 1] = preds[:, 0] & preds[:, 1]
    preds[:, 2] = preds[:, 0] & preds[:, 2]

    for i, cat in enumerate(["HS", "TR", "AG"]):
        cat_labels, cat_preds = labels[:, i], preds[:, i]

        precision, recall, f1, _ = precision_recall_fscore_support(
            cat_labels, cat_preds, average='binary', zero_division=0,
        )

        f1s.append(f1)
        precs.append(precision)
        recalls.append(recall)

        ret[cat.lower()+"_f1"] = f1
        ret[cat.lower()+"_precision"] = precision
        ret[cat.lower()+"_recall"] = recall

    neg_hs_f1_score = f1_score(1-(preds[:, 0] > 0), 1 - labels[:, 0])

    ret["macro_hs_f1_score"] = (f1s[0] + neg_hs_f1_score) / 2
    #
    # We calculate EMR in a gated way
    # Block TR and AG if HS is False
    #
    ret["emr_no_gating"] = accuracy_score(labels, original_preds)
    ret["emr"] = accuracy_score(labels, preds)

    ret["macro_f1"] = torch.Tensor(f1s).mean()
    ret["macro_precision"] = torch.Tensor(precs).mean()
    ret["macro_recall"] = torch.Tensor(recalls).mean()

    return ret


def get_task_b_metrics(predictions):

    outputs = predictions.predictions
    labels = predictions.label_ids

    return _get_b_metrics(outputs > 0, labels)
# Maps combinations to classes


combinatorial_mapping = {
    (0, 0, 0): 0,  # not hateful
    (1, 0, 0): 1,  # hs, not tr, not ag
    (1, 0, 1): 2,  # hs, not tr, ag
    (1, 1, 0): 3,  # hs, tr    , not ag
    (1, 1, 1): 4,  # hs, tr    , ag
}

inverse_combinatorial_mapping = {
    v: k for k, v in combinatorial_mapping.items()}


def get_combinatorial_metrics(predictions):
    outputs = predictions.predictions
    labels = predictions.label_ids

    preds = outputs.argmax(1)

    normalized_preds = np.array(
        [inverse_combinatorial_mapping[k] for k in preds])
    normalized_labels = np.array(
        [inverse_combinatorial_mapping[k] for k in labels])
    return _get_b_metrics(normalized_preds, normalized_labels)


class HierarchicalTrainer(Trainer):
    """
    Hierarchical Cross Entropy loss
    """

    def __init__(self, gamma=0, *args, **kwargs):
        """
        Gamma is the hyperparameter of this loss

        B(y) = (1 - y) γ + y * 1

        such that

        L(y, ypred) = L(y_HS, ypred_HS) + B(y_HS) (L(y_TR, ypred_TR) + L(y_AG, ypred_AG))

        If γ = 1, this is equal to standard sum of binary cross-entropies
        If equals to zero, it only sums the losses of the second-stage variables if the previous is one
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        unmasked_loss = loss_fct(logits, labels)
        # Expand to two columns, as this is the dimension of the second stage
        mask = labels[:, 0].view(-1, 1).expand(labels.shape[0], 2).clone()
        # Esto está mal porque floats pero bueno
        mask[mask < 1] = self.gamma

        first_stage_loss = unmasked_loss[:, 0]
        # Mask only the second stage
        second_stage_loss = (unmasked_loss[:, 1:] * mask).sum(1)
        loss = (first_stage_loss + second_stage_loss).sum()
        return (loss, outputs) if return_outputs else loss


def get_trainer_class(hierarchical=False, gamma=.0):
    """

    """
    if hierarchical:
        return (lambda *args, **kwargs:
                HierarchicalTrainer(*
                                    args, gamma=gamma, **kwargs)) if hierarchical else None


def get_metrics_fun(task_b, combinatorial):
    """
    Returns the function that computes the metrics
    """
    if task_b:
        if combinatorial:
            return get_combinatorial_metrics
        else:
            return get_task_b_metrics
    else:
        return None


def accepts(lang, **kwargs):
    """
    Returns whether the task is defined for the given language
    """
    return lang in ["it", "en", "es", "pt"]


def get_id2label(lang, task_b, combinatorial):
    """
    Returns a dictionary that maps the label id to the label name
    """
    if lang == "it":
        return {
            0: "hateful",
            1: "stereotype",
        }
    elif lang == "pt":
        return dict(enumerate(pt_labels))
    elif task_b:
        if combinatorial:
            return {
                0: "not hateful",
                1: "hateful, not tr, not ag",
                2: "hateful, not tr, ag",
                3: "hateful, targeted, not ag",
                4: "hateful, targeted, ag",
            }
        else:
            return {
                0: "hateful",
                1: "targeted",
                2: "aggressive",
            }
    else:
        return {
            0: 'ok',
            1: 'hateful',
        }


def train(
    base_model, lang, task_b=True, class_weight=None,
    hierarchical=False, gamma=.0, dev=False,
    combinatorial=False, use_defaults_if_not_tuned=False, **kwargs,
):
    """
    Train function

    Arguments:
    ---------

    task_b: bool (default False)
        If true, trains model for task_b

    combinatorial: bool (default False)
        If task_b true, whether to train a model in a combinatorial fashion

        That is, instead of training a different output for each predicted label,
        train a classifier for 5 possible combinations:
            0: not hateful
            1: HS, not TR, not AG
            2: HS, not TR, AG
            3: HS, TR, not AG
            4: HS, TR, AG
    """

    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(base_model, lang=lang))

    if dev:
        ds["test"] = ds["dev"]

    trainer_class = get_trainer_class(hierarchical, gamma)
    metrics_fun = get_metrics_fun(
        task_b=task_b, combinatorial=combinatorial) if lang not in {"it", "pt"} else None
    id2label = get_id2label(lang=lang, task_b=task_b,
                            combinatorial=combinatorial)

    if class_weight:
        class_weight = torch.Tensor([ds["train"][k]
                                    for k in ds["train"].features["label"].names])
        class_weight = 1 / (2 * class_weight.mean(1))

    training_args = get_training_arguments(base_model, task_name=task_name, lang=lang,
                                           metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned)

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        lang=lang, training_args=training_args,
        class_weight=class_weight, metrics_fun=metrics_fun, trainer_class=trainer_class,
        **kwargs
    )


def hp_tune(model_name, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    if lang == "it":
        id2label = {
            0: "hateful",
            1: "stereotype",
        }

        compute_metrics = None

    elif lang == "pt":
        id2label = dict(enumerate(pt_labels))
        compute_metrics = None
    else:
        id2label = {
            0: "hateful",
            1: "targeted",
            2: "aggressive",
        }
        compute_metrics = get_task_b_metrics

    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(
            model_name, lang=lang)
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
        compute_metrics=compute_metrics,
        config_info=config_info,
        **kwargs
    )
