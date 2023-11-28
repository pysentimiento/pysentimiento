import sys
import fire
import logging
import time
import wandb
import pandas as pd
from pysentimiento.config import config
from transformers.trainer_utils import set_seed
import pysentimiento.hate
import pysentimiento.sentiment
import pysentimiento.emotion
import pysentimiento.irony
import pysentimiento.lince.ner
import pysentimiento.targeted_sa
import pysentimiento.context_hate

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


modules = {
    "hate_speech": pysentimiento.hate,
    "sentiment": pysentimiento.sentiment,
    "emotion": pysentimiento.emotion,
    "irony": pysentimiento.irony,
    "ner": pysentimiento.lince.ner,
    "targeted_sa": pysentimiento.targeted_sa,
    "context_hate_speech": pysentimiento.context_hate,
}


def get_mean_performance(model, task, lang):
    df_results = pd.read_csv(
        f"data/results_{lang}.csv").set_index(["model", "task"])

    mean_macro_f1 = df_results.loc[(model, task), "mean macro f1"]

    return mean_macro_f1


def push_model(trainer, test_results, model, task, lang, push_to, ask_to_push=True):
    """
    Push model to huggingface
    """
    try:
        mean_macro_f1 = get_mean_performance(model, task, lang)
        print(f"Results for {model} at {task} ({lang})")
        print(f"Mean macro f1: {mean_macro_f1:.2f}")

    # File does not exist
    except FileNotFoundError as e:
        print(f"No results file found {e}")
        mean_macro_f1 = None
    except KeyError as e:
        print(f"Model {model} and {task} not found in results file")
        mean_macro_f1 = None

    model_macro_f1 = test_results.metrics["test_macro_f1"] * 100
    print(f"Model macro f1: {model_macro_f1:.2f}")

    trainer.args.overwrite_output_dir = True
    # Change output to current push_to
    trainer.args.output_dir = push_to

    push = False
    if mean_macro_f1 and model_macro_f1 > mean_macro_f1:
        print("Model macro f1 is better than average. Our Pushing model")
        # Push model
        push = True
    elif mean_macro_f1:
        print("Model macro f1 is lower than average.")

        if ask_to_push:
            res = input("Do you want to push the model anyway? (y/n)")
            if res == "y":
                push = True
    else:
        print("No mean macro f1 found to compare with")
        res = input("Do you want to push the model anyway? (y/n)")
        if res == "y":
            push = True

    if push:
        print(f"Pushing model to {push_to}")
        trainer.model.push_to_hub(push_to)
        trainer.tokenizer.push_to_hub(push_to)
        # Exit with success
        sys.exit(0)
    else:
        print("Not pushing model")
        # Return with error
        sys.exit(1)


def get_wandb_run_info(base_model, task, lang, **kwargs):
    # Check if task module has a get_wandb_run_info method

    if hasattr(modules[task], "get_wandb_run_info"):
        return modules[task].get_wandb_run_info(base_model, task, lang, **kwargs)
    else:
        return {
            "project": config["WANDB"]["PROJECT"],
            # Group by model name
            "group": f"{task}-{lang}",
            "job_type": f"{task}-{lang}-{base_model.split('/')[-1]}",
            # Name run by model name
            "config": {
                "model": base_model,
                "task": task,
                "lang": lang,
            }
        }


def train(
    base_model, task=None, lang="es",
    output_path=None,
    benchmark=False, times=10,
    push_to=None,
    limit=None, **kwargs
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
    if task is None:
        logger.error(f"Must provide task")
        sys.exit(1)

    if task is not None and task not in modules:
        logger.error(
            f"task ({task} was provided) must be one of {list(modules.keys())}")
        sys.exit(1)

    if task and not modules[task].accepts(lang):
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
        set_seed(int(time.time()))
        task_fun = modules[task].train

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

        for i in range(times):
            logger.info(f"{i+1} Iteration")
            # if wandb configured

            set_seed(int(time.time()))
            logger.info(
                f"Training {base_model} for {task} in lang {lang}")

            """
            Initialize Wandb
            """
            wandb_run = None
            try:
                wandb_run_info = get_wandb_run_info(
                    base_model, task, lang, **kwargs
                )
                wandb_run = wandb.init(
                    reinit=True,
                    **wandb_run_info
                )

                train_args["report_to"] = "wandb"
            except KeyError as e:
                logger.info(f"WANDB not configured. Skipping")

            task_fun = modules[task].train
            trainer, test_results = task_fun(
                base_model, lang=lang,
                **train_args
            )

            metrics = test_results.metrics

            if wandb_run:
                for k, v in metrics.items():
                    wandb.log({k: v})

            wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(train)
