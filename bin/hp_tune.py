import json
import sys
import fire
import os
import logging
import time
import wandb
from pysentimiento.data import load_fun, tasks
from pysentimiento.hate import hp_tune

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def hyperparameter_tune(
    model, task, lang,
    **kwargs
):
    """
    Hyperparameter tuning

    Args:
        model (str): Base model to use. Must be a HuggingFace model
        task (str): Task to train
        lang (str): Language to train



    """

    if task not in tasks:
        logger.error(
            f"task ({task} was provided) must be one of {list(load_fun.keys())}")
        sys.exit(1)

    if task and lang not in load_fun[task]:
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    # Load model

    hp_tune(model, lang)


if __name__ == "__main__":
    fire.Fire(hyperparameter_tune)
