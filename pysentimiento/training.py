import torch
import os
import torch
import tempfile
import logging
from .metrics import compute_metrics
from .config import config
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer
)
from .preprocessing import special_tokens, get_preprocessing_args


logging.basicConfig()
logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def load_model(
    base_model, id2label, lang, max_length=128, auto_class=AutoModelForSequenceClassification
):
    """
    Loads model and tokenizer
    """
    logger.debug(f"Loading model {base_model}")
    model = auto_class.from_pretrained(
        base_model, return_dict=True, num_labels=len(id2label)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = max_length

    if type(id2label) is not dict:
        id2label = {str(i): label for i, label in enumerate(id2label)}
    label2id = {label: i for i, label in id2label.items()}

    model.config.id2label = id2label
    model.config.label2id = label2id

    # Add special tokens
    special_tokens = list(get_preprocessing_args(base_model, lang).values())

    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


class MultiLabelTrainer(Trainer):
    """
    Multilabel and class weighted trainer
    """

    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.model.config.problem_type == "multi_label_classification":
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_huggingface(
        base_model, dataset, id2label,
        metrics_fun, training_args, lang,
        max_length=128, auto_class=AutoModelForSequenceClassification,
        format_dataset=None, use_dynamic_padding=True, class_weight=None, trainer_class=None,  data_collator_class=DataCollatorWithPadding, tokenize_fun=None,
        **kwargs):
    """
    Run experiments experiments
    """
    padding = False if use_dynamic_padding else 'max_length'

    model, tokenizer = load_model(
        base_model, id2label=id2label, lang=lang,
        max_length=max_length, auto_class=auto_class,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    def _tokenize_fun(x):
        if tokenize_fun:
            return tokenize_fun(x)
        else:
            return tokenizer(x['text'], padding=padding, truncation=True)

    dataset = dataset.map(
        _tokenize_fun, batched=True
    )

    if use_dynamic_padding:
        data_collator = data_collator_class(tokenizer, padding="longest")
    else:
        if not format_dataset:
            raise ValueError(
                "Must provide format_dataset if not using dynamic padding")

    if format_dataset:
        for split in dataset.keys():
            dataset[split] = format_dataset(dataset[split])

    try:
        tmp_path = config["PYSENTIMIENTO"]["TMP_DIR"]
    except KeyError:
        tmp_path = None

    output_path = tempfile.mkdtemp(
        prefix="pysentimiento",
        dir=tmp_path
    )

    trainer_args = {
        "model": model,
        "args": training_args,
        "compute_metrics": metrics_fun,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["dev"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }

    if class_weight is not None:
        class_weight = class_weight.to(device)
        print(f"Using class weight = {class_weight}")
        trainer_class = MultiLabelTrainer
        trainer_args["class_weight"] = class_weight
    else:
        trainer_class = trainer_class or Trainer

    trainer = trainer_class(**trainer_args)

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    os.system(f"rm -Rf {output_path}")

    return trainer, test_results


def train_and_eval(base_model, dataset, id2label,
                   lang, limit=None, metrics_fun=None,
                   **kwargs):
    """
    Base function
    """
    if limit:
        """
        Smoke test
        """
        dataset = dataset.select(range(limit))

    if type(id2label) is list:
        id2label = {i: label for i, label in enumerate(id2label)}

    label2id = {v: k for k, v in id2label.items()}

    if not metrics_fun:
        def metrics_fun(x): return compute_metrics(x, id2label=id2label)

    if base_model == "rnn":
        # TODO: Fix this
        # Remove baselines from here! and don't use torchtext :)
        from .baselines.training import train_rnn_model, train_ffn_model
        return train_rnn_model(
            dataset["train"], dataset["dev"], dataset["test"], lang=lang, id2label=id2label, metrics_fun=metrics_fun,
            **kwargs
        )
    elif base_model == "ffn":
        # TODO: Fix this
        # Remove baselines from here! and don't use torchtext :)
        from .baselines.training import train_rnn_model, train_ffn_model
        return train_ffn_model(
            dataset["train"], dataset["dev"], dataset["test"], lang=lang, id2label=id2label, metrics_fun=metrics_fun,
            **kwargs
        )
    else:
        """
        Transformer classifier
        """

        return train_huggingface(
            base_model=base_model, dataset=dataset, id2label=id2label,
            lang=lang, metrics_fun=metrics_fun, **kwargs
        )
