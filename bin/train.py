import sys
import fire
import torch
import logging
from pysentimiento.hate import train as train_hate
from sklearn.utils.class_weight import compute_class_weight


"""
Training functions
"""
train_fun = {
    "hate_speech": {
        "es": train_hate,
        "en": train_hate,
    }
}

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def train(
    base_model, output_path, task, lang="es",
    epochs=5, batch_size=32, eval_batch_size=16,
    warmup_ratio=.1, limit=None, **kwargs
):
    """
    Script to train models

    Arguments:
    =========

    base_model: str
        Huggingface's model identifier or path to model

    output_path: str
        Where to save the trained model

    task: str
        One of 'sentiment', 'emotion', 'hate_speech'

    """

    if task not in train_fun:
        logger.error(f"task ({task} was provided) must be one of {list(train_fun.keys())}")
        sys.exit(1)

    if lang not in train_fun[task]:
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    logger.info(kwargs)
    logger.info(f"Training {base_model} for {task} in lang {lang}")

    task_fun = train_fun[task][lang]

    trainer, test_results = task_fun(
        base_model, lang,
        epochs=epochs, batch_size=batch_size, eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio, limit=limit,
        **kwargs
    )

    logger.info("Test results")
    logger.info("=" * 50)
    for k, v in test_results.items():
        print(f"{k:<16} : {v:.3f}")


    logger.info(f"Saving model to {output_path}")
    trainer.save_model(output_path)

if __name__ == "__main__":
    fire.Fire(train)
