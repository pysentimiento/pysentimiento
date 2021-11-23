
import torch
import pytorch_lightning as pl
import fasttext
import unidecode
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets
from torchtext.data.utils import get_tokenizer
from pysentimiento.baselines.models import FFNModel, RNNModel
from pysentimiento.baselines.utils import build_vocab, build_embedding_matrix

class EvalResults(object):
    def __init__(self, predictions, metrics):
        self.predictions = predictions
        self.metrics = metrics

    def __str__(self):
        return f"\n{self.metrics}"


def tokenize_dataset(dataset, tokenizer, vocab):
    """
    Tokenize and format a dataset
    """
    stoi = vocab.get_stoi()
    def tokenize(batch):
        text = unidecode.unidecode(batch['text'].lower())
        tokens = tokenizer(text)

        token_ids = [stoi[t] for t in tokens]
        return {"input_ids": token_ids}

    return dataset.map(tokenize, batched=False)

def format(dataset):
    """
    Format dataset
    """
    dataset = dataset.map(lambda examples: {'labels': examples['label']})
    return dataset

def get_dataloader(dataset, batch_size, pad_idx):
    """
    Get a dataloader for a dataset
    """

    def collate_batch(batch):

        labels = [t["labels"].tolist() if type(t["labels"]) is torch.Tensor else t["labels"] for t in batch]
        input_ids = [t["input_ids"] for t in batch]
        # Return text, text_lens, labels
        text = pad_sequence(input_ids, padding_value=pad_idx, batch_first=True)
        lens = torch.tensor([len(t) for t in input_ids])
        labels = torch.tensor(labels)
        return text, lens, labels

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)

def build_dataloaders(train_dataset, dev_dataset, test_dataset, tokenizer, vocab, batch_size, format_dataset=None):
    """
    Build dataloaders
    """

    train_dataset = tokenize_dataset(train_dataset, tokenizer, vocab)
    dev_dataset = tokenize_dataset(dev_dataset, tokenizer, vocab)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, vocab)

    if format_dataset:
        train_dataset = format_dataset(train_dataset)
        dev_dataset = format_dataset(dev_dataset)
        test_dataset = format_dataset(test_dataset)
    else:
        train_dataset = format(train_dataset)
        dev_dataset = format(dev_dataset)
        test_dataset = format(test_dataset)


    for dataset in [train_dataset, dev_dataset, test_dataset]:
        dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    PAD_IDX = vocab.get_stoi()["<pad>"]


    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, pad_idx=PAD_IDX)
    dev_dataloader = get_dataloader(dev_dataset, batch_size=batch_size, pad_idx=PAD_IDX)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, pad_idx=PAD_IDX)

    return train_dataloader, dev_dataloader, test_dataloader


def train_rnn_model(train_dataset, dev_dataset, test_dataset, lang, id2label, embeddings_path, batch_size=32, epochs=5, bidirectional=False, rnn_units=512, format_dataset=None, **kwargs):
    """
    Train an RNN model
    """

    tokenizer = get_tokenizer("spacy", lang)
    vocab = build_vocab(concatenate_datasets([train_dataset, dev_dataset, test_dataset]), tokenizer)

    PAD_IDX = vocab.get_stoi()["<pad>"]

    train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(train_dataset, dev_dataset, test_dataset, tokenizer, vocab, batch_size, format_dataset)

    emb_matrix = build_embedding_matrix(vocab, fasttext.load_model(embeddings_path))

    model = RNNModel(
        vocab_size=len(vocab), embedding_dim=emb_matrix.shape[-1], pad_idx=PAD_IDX, rnn_units=rnn_units, bidirectional=bidirectional, id2label=id2label,
        embedding_matrix=emb_matrix, freeze_embeddings=True, num_labels=len(id2label),
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1
    )
    trainer.fit(model, train_dataloader, dev_dataloader)
    results = EvalResults(
        predictions=torch.cat(trainer.predict(model, test_dataloader)),
        metrics=trainer.test(model, test_dataloader)[0]
    )
    return trainer, results

def train_ffn_model(train_dataset, dev_dataset, test_dataset, lang, id2label, embeddings_path, batch_size=32, epochs=5, hidden_units=512, format_dataset=None, **kwargs):
    """
    Train an RNN model
    """

    tokenizer = get_tokenizer("spacy", lang)
    vocab = build_vocab(concatenate_datasets([train_dataset, dev_dataset, test_dataset]), tokenizer)

    PAD_IDX = vocab.get_stoi()["<pad>"]

    train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(train_dataset, dev_dataset, test_dataset, tokenizer, vocab, batch_size, format_dataset)

    emb_matrix = build_embedding_matrix(vocab, fasttext.load_model(embeddings_path))

    model = FFNModel(
        vocab_size=len(vocab), embedding_dim=emb_matrix.shape[-1], pad_idx=PAD_IDX,
        hidden_units=hidden_units,
        embedding_matrix=emb_matrix, freeze_embeddings=True, num_labels=len(id2label),
        id2label=id2label,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1
    )
    trainer.fit(model, train_dataloader, dev_dataloader)
    results = EvalResults(
        predictions=torch.cat(trainer.predict(model, test_dataloader)),
        metrics=trainer.test(model, test_dataloader)[0]
    )
    return trainer, results

