import json
import sys
import fire
import os
import logging
import time
import transformers
from pysentimiento.hate import train as train_hate
from pysentimiento.sentiment import train as train_sentiment
from pysentimiento.emotion import train as train_emotion
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
    }
}

lang_fun = {
    lang: {} for lang in ["es", "en"]
}

for lang in lang_fun:
    for task, task_funs in train_fun.items():
        if lang in task_funs:
            lang_fun[lang][task] = task_funs[lang]



logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def train(
    base_model, task, lang="es",
    output_path=None,
    benchmark=False, times=10, benchmark_output_path=None,
    epochs=5, batch_size=32, eval_batch_size=16,
    warmup_ratio=.1, limit=None, predict=False, overwrite=False, **kwargs
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
    """
    if task not in train_fun:
        logger.error(f"task ({task} was provided) must be one of {list(train_fun.keys())}")
        sys.exit(1)

    if lang not in train_fun[task]:
        logger.error(f"Lang {lang} not available for {task}")
        sys.exit(1)

    if benchmark and not benchmark_output_path:
        logger.error(f"Must provide benchmark_output_path in benchmark mode")
        sys.exit(1)

    logger.info(kwargs)

    task_fun = train_fun[task][lang]
    train_args = {
        **{
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "warmup_ratio": warmup_ratio,
            "limit": limit,
        },
        **kwargs
    }

    if not benchmark:
        """
        Training!
        """
        logger.info(f"Training {base_model} for {task} in lang {lang}")


        trainer, test_results = task_fun(
            base_model, lang,
            **train_args
        )
        logger.info("Test results")
        logger.info("=" * 50)
        for k, v in test_results.metrics.items():
            print(f"{k:<16} : {v:.3f}")


        logger.info(f"Saving model to {output_path}")
        trainer.save_model(output_path)

    else:
        """
        Benchmark mode
        """
        logger.info(f"Benchmarking {base_model} for {task} in {lang}")
        tasks = [task] if task else lang_fun[lang].keys()

        if os.path.exists(benchmark_output_path) and not overwrite:
            with open(benchmark_output_path, "r") as f:
                results = json.load(f)

            results[task] = []
        else:

            results = {
                "model": base_model,
                "lang": lang,
                "train_args": train_args,
                "evaluations": {k: [] for k in tasks},
            }

            if predict:
                results["predictions"] = {k: [] for k in tasks}

        logger.info(results)

        for i in range(times):
            logger.info(f"{i+1} Iteration")

            for task_name in tasks:
                set_seed(int(time.time()))
                logger.info(f"Training {base_model} for {task_name} in lang {lang}")
                task_fun = lang_fun[lang][task_name]
                trainer, test_results = task_fun(
                    base_model, lang,
                    **train_args
                )


                if predict:
                    results["predictions"][task_name].append(test_results.predictions.tolist())

                metrics = test_results.metrics
                results["evaluations"][task_name].append(metrics)

                logger.info("Test results")
                logger.info("=" * 50)
                for k, v in metrics.items():
                    logger.info(f"{k:<16} : {v:.3f}")

                with open(benchmark_output_path, "w+") as f:
                    json.dump(results, f, indent=4)
        logger.info(f"{times} runs of {tasks} saved to {benchmark_output_path}")

if __name__ == "__main__":
    fire.Fire(train)