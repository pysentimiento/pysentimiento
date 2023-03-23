import sys
import fire
import logging
import pysentimiento.hate
import pysentimiento.sentiment
import pysentimiento.emotion
import pysentimiento.irony
import pysentimiento.targeted_sa

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


modules = {
    "hate_speech": pysentimiento.hate,
    "sentiment": pysentimiento.sentiment,
    "emotion": pysentimiento.emotion,
    "irony": pysentimiento.irony,
    "targeted_sa": pysentimiento.targeted_sa,
}


def hyperparameter_tune(
    model, task, lang, count=42,
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

    if task not in modules:
        logger.error(
            f"task ({task} was provided) must be one of {list(modules.keys())}")
        sys.exit(1)

    if not modules[task].accepts(lang):
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    # Run hp tune
    modules[task].hp_tune(model, lang, count=count, **kwargs)


if __name__ == "__main__":
    fire.Fire(hyperparameter_tune)
