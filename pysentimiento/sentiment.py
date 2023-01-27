import wandb
from .tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
)
from .training import train_model, load_model
from .config import config
from .semeval import (
    load_datasets as load_semeval_datasets,
    id2label as id2labelsemeval, label2id as label2idsemeval
)
from .metrics import compute_metrics
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

lang_conf = {
    "es": {
        "load_datasets": load_tass_datasets,
        "id2label": id2labeltass,
        "label2id": label2idtass,
    },
    "en": {
        "load_datasets": load_semeval_datasets,
        "id2label": id2labelsemeval,
        "label2id": label2idsemeval,
    }
}

extra_args = {
    "vinai/bertweet-base": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}


def load_datasets(lang, **kwargs):
    """
    """
    return lang_conf[lang]["load_datasets"](lang=lang, **kwargs)


def train(
    base_model, lang="es", epochs=5, batch_size=32,
    limit=None, **kwargs
):
    """
    """
    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]

    load_extra_args = extra_args[base_model] if base_model in extra_args else {
    }

    ds = load_datasets(**load_extra_args)

    kwargs = {
        **kwargs,
        **{
            "id2label": id2label,
            "epochs": epochs,
            "batch_size": batch_size,
            "limit": limit,
            "lang": lang,
        }
    }

    return train_model(base_model, ds["train"], ds["dev"], ds["test"], **kwargs)


def hp_tune(model_name, lang):
    """
    Hyperparameter tuning with wandb
    """
    task_name = "sentiment"

    # method
    sweep_config = {
        'method': 'random',
    }
    # hyperparameters
    parameters_dict = {
        'epochs': {
            'value': 1
        },
        'batch_size': {
            'values': [8, 16, 32, 64]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'weight_decay': {
            'values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
        },

        'warmup_ratio': {
            'values': [0.06, 0.08, 0.10]
        }
    }

    sweep_config['parameters'] = parameters_dict
    id2label = lang_conf[lang]["id2label"]
    ds = load_datasets(lang=lang)

    def model_init():
        model, _ = load_model(model_name, id2label)
        return model

    _, tokenizer = load_model(model_name, id2label)

    tokenized_ds = ds.map(
        lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True), batched=True, batch_size=32)

    tokenized_ds = tokenized_ds.remove_columns(["text", "lang"])

    def train(config=None):
        init_params = {
            "config": config,
            "group": f"sweep-{task_name}-{lang}",
            "job_type": f"{task_name}-{lang}-{model_name}",
            "config": {
                "model": model_name,
                "task": task_name,
                "lang": lang,
            },
        }
        with wandb.init(**init_params):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = TrainingArguments(
                output_dir='./tmp/sweeps',
                report_to='wandb',  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                warmup_ratio=config.warmup_ratio,
                per_device_eval_batch_size=16,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
                remove_unused_columns=False,
            )

            # define training loop
            trainer = Trainer(
                # model,
                model_init=model_init,
                args=training_args,
                compute_metrics=lambda x: compute_metrics(x, id2label),
                train_dataset=tokenized_ds['train'],
                eval_dataset=tokenized_ds['dev'],
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(
                    tokenizer, padding="longest"),
            )

            # start training loop
            trainer.train()

    # Initiate sweep
    sweep_id = wandb.sweep(sweep_config, project=config["WANDB"]["PROJECT"])

    wandb.agent(sweep_id, train)
