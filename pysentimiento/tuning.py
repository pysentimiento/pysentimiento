import wandb
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from .config import config
from .metrics import compute_metrics as _compute_metrics

# hyperparameters
parameters_dict = {
    'epochs': {
        'values': [3, 4, 5],
    },
    'batch_size': {
        'value': 32
    },
    'learning_rate': {
        'values': [2e-5, 3e-5, 5e-5, 6e-5, 7e-5, 8e-5, 1e-4]
    },

    'weight_decay': {
        'value': 0.1
    },

    'warmup_ratio': {
        'values': [0.06, 0.08, 0.10]
    },
}


def hyperparameter_sweep(
        name, group_name, model_init, tokenizer, datasets, id2label, sweep_method="random", format_dataset=None, compute_metrics=None, config_info=None,
        metric_for_best_model="eval_macro_f1", count=None):
    """
    Hyperparameter sweep with wandb

    Args:

        model_init (function): function that returns a model
        tokenizer (transformers.PreTrainedTokenizer): tokenizer
        datasets (DatasetDict): dataset dictionary
        id2label (dict): dictionary with id to label mapping

        sweep_method (str): sweep method, e.g. random, bayesian, grid, etc. Defaults to "random".
        format_dataset (function, optional): function to format dataset. Defaults to None.
        compute_metrics (function, optional): function to compute metrics. Defaults to None.
        config_info (dict, optional): config info. Defaults to None.
    """
    sweep_config = {
        'name': name,
        'method': sweep_method,
        'parameters': parameters_dict,
    }

    if compute_metrics is None:
        def compute_metrics(preds): return _compute_metrics(preds, id2label)

    tokenized_ds = datasets.map(
        lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True), batched=True, batch_size=32)

    if format_dataset is not None:
        tokenized_ds = tokenized_ds.map(format_dataset)

    tokenized_ds = tokenized_ds.remove_columns([
        x for x in datasets['train'].column_names if x not in ['labels', 'label']
    ])

    def train(config=None):

        init_params = {
            "config": config or {},
            "group": group_name,
            "job_type": "sweep",
        }

        if config_info:
            init_params["config"].update(config_info)

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
                per_device_eval_batch_size=config.batch_size,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                remove_unused_columns=False,
                group_by_length=True,
            )

            # define training loop
            trainer = Trainer(
                # model,
                model_init=model_init,
                args=training_args,
                compute_metrics=compute_metrics,
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

    wandb.agent(sweep_id, train, count=count)