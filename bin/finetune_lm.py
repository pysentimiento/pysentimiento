import os
import fire
import torch
from glob import glob
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def finetune_lm(
    model_name, path_to_tweets, epochs=3, eval_steps=1000, eval_files_ratio=0.95,
    base_model_name='dccuchile/bert-base-spanish-wwm-cased', max_length=128,
    file_limit=None, batch_size=48, eval_batch_size=16
    ):
    """
    Finetune LM on tweets
    """
    print(f"Finetuning {model_name}")
    print(f"Max length: {max_length}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForMaskedLM.from_pretrained(base_model_name, return_dict=True)
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
    tokenizer.model_max_length = max_length


    tweet_files = glob(os.path.join(path_to_tweets, "*.txt"))

    if file_limit:
        tweet_files = tweet_files[:file_limit]
    limit = int(eval_files_ratio * len(tweet_files))

    train_files = tweet_files[:limit]
    dev_files = tweet_files[limit:]

    print(f"Training with {limit} files -- testing with {len(dev_files)}")

    train_dataset, test_dataset = load_dataset("text", data_files={"train": train_files, "test": dev_files}, split=["train", "test"])

    print(f"Train: {len(train_dataset):e} instances")
    print(f"Dev: {len(test_dataset):e} instances")

    print("\nTokenizing...")
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    model_path = f"./models/{model_name}"

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        save_steps=2000,
        do_eval= True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    print("Saving...")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == '__main__':
    fire.Fire(finetune_lm)
