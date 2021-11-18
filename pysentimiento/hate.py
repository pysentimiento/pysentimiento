
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
from transformers import Trainer
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
    (0, 0, 0): 0, # not hateful
    (1, 0, 0): 1, # hs, not tr, not ag
    (1, 0, 1): 2, # hs, not tr, ag
    (1, 1, 0): 3, # hs, tr    , not ag
    (1, 1, 1): 4, # hs, tr    , ag
}

inverse_combinatorial_mapping = {v:k for k, v in combinatorial_mapping.items()}

def get_combinatorial_metrics(predictions):
    outputs = predictions.predictions
    labels = predictions.label_ids

    preds = outputs.argmax(1)

    normalized_preds = np.array([inverse_combinatorial_mapping[k] for k in preds])
    normalized_labels = np.array([inverse_combinatorial_mapping[k] for k in labels])
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




def train(
    base_model, lang, epochs=5, batch_size=32,
    warmup_ratio=.1, limit=None, accumulation_steps=1, task_b=True, class_weight=None,
    hierarchical=False, gamma=.0, dev=False, metric_for_best_model="macro_f1",
    combinatorial=False, **kwargs,
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

    train_dataset, dev_dataset, test_dataset = load_datasets(
        lang=lang,
        preprocessing_args=extra_args.get(base_model, {})
    )


    if dev:
        test_dataset = dev_dataset

    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))

    trainer_class = None
    metrics_fun = None
    if task_b:
        if combinatorial:
            id2label = {
                0: "not hateful",
                1: "hateful, not tr, not ag",
                2: "hateful, not tr, ag",
                3: "hateful, targeted, not ag",
                4: "hateful, targeted, ag",
            }

            metrics_fun = get_combinatorial_metrics
        else:
            metrics_fun = get_task_b_metrics
            id2label = {
                0: "hateful",
                1: "targeted",
                2: "aggressive",
            }

        trainer_class = (lambda *args, **kwargs: HierarchicalTrainer(*args, gamma=gamma, **kwargs)) if hierarchical else None
    else:
        metrics_fun = None
        id2label = {
            0: 'ok',
            1: 'hateful',
        }


    label2id = {v:k for k, v in id2label.items()}


    problem_type = "multi_label_classification" if (task_b and not combinatorial) else "single_label_classification"

    labels_order = ["HS", "TR", "AG"]
    def format_dataset(dataset):
        def get_labels(examples):
            if task_b:
                if combinatorial:
                    # Convert to a single label
                    return {'labels': combinatorial_mapping[
                        tuple(examples[k] for k in labels_order)
                    ]}
                return {'labels': torch.Tensor([examples[k] for k in labels_order])}
            else:
                return {'labels': examples["HS"]}
        dataset = dataset.map(get_labels)
        return dataset

    if class_weight:
        class_weight = torch.Tensor([train_dataset[k] for k in labels_order])
        class_weight = 1 / (2* class_weight.mean(1))


    return train_model(
        base_model, train_dataset, dev_dataset, test_dataset, id2label,
        format_dataset=format_dataset, lang=lang,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
        warmup_ratio=warmup_ratio, accumulation_steps=accumulation_steps,
        metrics_fun=metrics_fun, trainer_class=trainer_class, metric_for_best_model=metric_for_best_model,
        **kwargs
    )
