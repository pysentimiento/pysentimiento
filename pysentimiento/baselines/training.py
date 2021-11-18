
import torch
import pytorch_lightning as pl
import fasttext
import unidecode
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets
from torchtext.data.utils import get_tokenizer
from pysentimiento.baselines.models import RNNModel
from pysentimiento.baselines.utils import build_vocab, build_embedding_matrix

class EvalResults(object):
    def __init__(self, predictions, metrics):
        self.predictions = predictions
        self.metrics = metrics

    def __str__(self):
        return f"\n{self.metrics}"

def train_rnn_model(train_dataset, dev_dataset, test_dataset, lang, id2label, embeddings_path, batch_size=32, epochs=5, **kwargs):
    """
    Train an RNN model
    """

    tokenizer = get_tokenizer("spacy", lang)
    vocab = build_vocab(concatenate_datasets([train_dataset, dev_dataset, test_dataset]), tokenizer)


    stoi = vocab.get_stoi()
    itos = vocab.get_itos()



    def tokenize(batch):
        text = unidecode.unidecode(batch['text'].lower())
        tokens = tokenizer(text)

        token_ids = [stoi[t] for t in tokens]
        return {"input_ids": token_ids}

    train_dataset = train_dataset.map(tokenize, batched=False)
    dev_dataset = dev_dataset.map(tokenize, batched=False)
    test_dataset = test_dataset.map(tokenize, batched=False)

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        dataset.set_format(type='torch', columns=['input_ids', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)

    emb_matrix = build_embedding_matrix(vocab, fasttext.load_model(embeddings_path))


    PAD_IDX = stoi["<pad>"]

    def collate_batch(batch):
        labels = [t["labels"] for t in batch]
        input_ids = [t["input_ids"] for t in batch]

        # Return text, text_lens, labels
        text = pad_sequence(input_ids, padding_value=PAD_IDX, batch_first=True)
        lens = torch.tensor([len(t) for t in input_ids])
        labels = torch.tensor(labels)
        return text, lens, labels


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNModel(
        vocab_size=len(vocab), embedding_dim=emb_matrix.shape[-1], pad_idx=PAD_IDX, rnn_units=256,
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