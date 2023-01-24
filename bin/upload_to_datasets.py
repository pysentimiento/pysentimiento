import fire
import logging
from pysentimiento.data import load_datasets
from datasets import DatasetDict

logging.basicConfig()

logger = logging.getLogger('pysentimiento')
logger.setLevel(logging.INFO)


def upload_to_datasets(task="all", lang="all"):
    """
    Upload datasets to huggingface's dataset hub
    """
    if task == "all":
        tasks = ["hate_speech", "sentiment", "emotion",
                 "irony", "ner", "pos", "lince_sentiment"]
    else:
        tasks = [task]

    if lang == "all":
        langs = ["es", "en"]
    else:
        langs = [lang]

    issues = []
    for task in tasks:
        for lang in langs:
            print(("="*80+"\n") * 3)
            try:
                ds = load_datasets(task, lang, preprocess=False)
                ds.push_to_hub(
                    f"pysentimiento/{lang}_{task}",
                    private=True
                )
                logger.info(f"Uploaded {task}-{lang} to datasets hub")
            except Exception as e:
                logger.error(f"Error with {task}-{lang}")
                logger.error(e)
                issues.append((task, lang, e))

    print("="*80)
    print("="*80)
    logger.error(f"Found {len(issues)} issues")
    for issue in issues:
        logger.error(issue)


if __name__ == "__main__":
    fire.Fire(upload_to_datasets)
