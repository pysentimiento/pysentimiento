import json
import sys
import fire
import os
import logging
import time
import wandb
import pysentimiento.hate
import pysentimiento.sentiment
import pysentimiento.emotion
import pysentimiento.irony
from pysentimiento.data import load_fun, tasks

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


modules = {
    "hate_speech": pysentimiento.hate,
    "sentiment": pysentimiento.sentiment,
    "emotion": pysentimiento.emotion,
    "irony": pysentimiento.irony,
}


def hyperparameter_tune(
    model, task, lang, count=None,
    **kwargs
):
    """
    Hyperparameter tuning

    Args:
        model (str): Base model to use. Must be a HuggingFace model
        task (str): Task to train
        lang (str): Language to train
        count (int): Number of runs to perform
    """

    if task not in tasks:
        logger.error(
            f"task ({task} was provided) must be one of {list(load_fun.keys())}")
        sys.exit(1)

    if task and lang not in load_fun[task]:
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    # Run hp tune
    modules[task].hp_tune(model, lang, count=count, **kwargs)


if __name__ == "__main__":
    fire.Fire(hyperparameter_tune)
