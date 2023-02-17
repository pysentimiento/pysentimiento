import sys
import fire
import logging
import time
import wandb
import pandas as pd
from pysentimiento.config import config
from pysentimiento.hate import train as train_hate
from pysentimiento.sentiment import train as train_sentiment
from pysentimiento.emotion import train as train_emotion
from pysentimiento.irony import train as train_irony
from pysentimiento.lince import train_ner, train_pos, train_sentiment as train_sentiment_lince
from transformers.trainer_utils import set_seed

"""
Training functions
"""
train_fun = {
    "hate_speech": {
        "es": train_hate,
        "en": train_hate,
    },
    "sentiment": {
        "es": train_sentiment,
        "en": train_sentiment,
    },
    "emotion": {
        "es": train_emotion,
        "en": train_emotion,
    },

    "irony": {
        "es": train_irony,
        "en": train_irony,
    },

    # We use multilingual LinCE dataset here
    "ner": {
        "es": train_ner,
        "en": train_ner,
    },

    # We use multilingual LinCE dataset here
    "pos": {
        "es": train_pos,
        "en": train_pos,
    },

    "lince_sentiment": {
        "es": train_sentiment_lince,
        "en": train_sentiment_lince,
    },
}

lang_fun = {
    lang: {} for lang in ["es", "en"]
}

for lang in lang_fun:
    for _task, task_funs in train_fun.items():
        if lang in task_funs:
            lang_fun[lang][_task] = task_funs[lang]


logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def push_model(trainer, test_results, model, task, lang, push_to, ask_to_push=True):
    """
    Push model to huggingface
    """
    df_results = pd.read_csv(
        "data/results.csv").set_index(["lang", "model", "task"])

    print(f"Results for {model} at {task} ({lang})")
    mean_macro_f1 = df_results.loc[(lang, model, task), "mean macro f1"]

    print(f"Mean macro f1: {mean_macro_f1:.2f}")

    model_macro_f1 = test_results.metrics["test_macro_f1"] * 100

    print(f"Model macro f1: {model_macro_f1:.2f}")

    trainer.args.overwrite_output_dir = True
    # Change output to current push_to
    trainer.args.output_dir = push_to

    push = False
    if model_macro_f1 > mean_macro_f1:
        print("Mean macro f1 is lower than model macro f1. Pushing model")
        # Push model
        push = True
    else:
        print("Mean macro f1 is higher than model macro f1.")

        if ask_to_push:
            res = input("Do you want to push the model anyway? (y/n)")
            if res == "y":
                push = True

    if push:
        print(f"Pushing model to {push_to}")
        trainer.model.push_to_hub(push_to)
        trainer.tokenizer.push_to_hub(push_to)
    else:
        print("Not pushing model")


def train(
    base_model, task=None, lang="es",
    output_path=None,
    benchmark=False, times=10,
    push_to=None, ask_to_push=True,
    limit=None, predict=False, **kwargs
):
    """
    Script to train models

    Arguments:
    =========

    base_model: str
        Huggingface's model identifier or path to model

    task: str
        One of 'sentiment', 'emotion', 'hate_speech'

    output: str, Optional
        Where to save the trained model

    benchmark: bool (default False)
        If true, train and evaluate n-times for the given task. Saves the evaluation in

        No model is saved after this benchmark

    times: int (default 10)
        If benchmark is true, this argument determines the number of times the

    push_to: str, Optional
        If provided, push the results to huggingface.
    """
    if task is None and not benchmark:
        logger.error(f"Must provide task if not in benchmark mode")
        sys.exit(1)

    if task is not None and task not in train_fun:
        logger.error(
            f"task ({task} was provided) must be one of {list(train_fun.keys())}")
        sys.exit(1)

    if task and lang not in train_fun[task]:
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    logger.info(kwargs)

    train_args = kwargs.copy()
    if limit:
        train_args["limit"] = limit

    if not benchmark:
        """
        Training!
        """

        task_fun = train_fun[task][lang]

        logger.info(f"Training {base_model} for {task} in lang {lang}")

        trainer, test_results = task_fun(
            base_model, lang, dont_report=True,
            **train_args
        )
        logger.info("Test results")
        logger.info("=" * 50)
        for k, v in test_results.metrics.items():
            print(f"{k:<16} : {v:.3f}")

        if push_to:
            push_model(
                trainer=trainer, test_results=test_results,
                model=base_model, task=task, lang=lang,
                push_to=push_to,
            )
        elif output_path:
            logger.info(f"Saving model to {output_path}")
            trainer.save_model(output_path)

    else:
        """
        Benchmark mode
        """
        logger.info(f"Benchmarking {base_model} for {task} in {lang}")

        tasks = [task] if task else lang_fun[lang].keys()

        for i in range(times):
            logger.info(f"{i+1} Iteration")
            # if wandb configured

            for task_name in tasks:
                set_seed(int(time.time()))
                logger.info(
                    f"Training {base_model} for {task_name} in lang {lang}")

                """
                Initialize Wandb
                """
                wandb_run = None
                try:
                    wandb_run = wandb.init(
                        project=config["WANDB"]["PROJECT"],
                        # Group by model name
                        group=f"{task_name}-{lang}",
                        job_type=f"{task_name}-{lang}-{base_model}",
                        # Name run by model name
                        config={
                            "model": base_model,
                            "task": task_name,
                            "lang": lang,
                        },
                        reinit=True,
                    )

                    train_args["report_to"] = "wandb"
                except KeyError as e:
                    logger.info(f"WANDB not configured. Skipping")

                task_fun = lang_fun[lang][task_name]
                trainer, test_results = task_fun(
                    base_model, lang,
                    **train_args
                )

                metrics = test_results.metrics

                if wandb_run:
                    for k, v in metrics.items():
                        wandb.log({k: v})

                wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(train)
