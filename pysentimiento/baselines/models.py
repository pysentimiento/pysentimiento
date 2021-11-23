import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pysentimiento.metrics import get_metrics


class BaseModel(pl.LightningModule):
    def __init__(self, id2label):
        super().__init__()
        self.problem_type = None
        self.id2label = id2label
        self.loss_fun = None

    def _set_loss_and_problem_type(self, y):
        if not self.problem_type:
            if y.dim() > 1 and y.shape[-1] > 1:
                self.problem_type = "multi_label_classification"
                self.loss_fun = F.binary_cross_entropy_with_logits
            else:
                self.problem_type = "single_label_classification"
                self.loss_fun = F.cross_entropy

    def training_step(self, batch, batch_idx):
        x, lens, y = batch

        self._set_loss_and_problem_type(y)

        outs = self.forward(x, lens)
        loss = self.loss_fun(outs, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lens, y = batch
        self._set_loss_and_problem_type(y)

        outs = self.forward(x, lens)
        loss = self.loss_fun(outs, y)
        preds = outs.cpu()
        metrics = get_metrics(preds, y.cpu(), self.id2label)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        for k, v in metrics.items():
            self.log("val_"+k, v, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, lens, y = batch
        outs = self.forward(x, lens)
        preds = outs.cpu()
        metrics = get_metrics(preds, y.cpu(), self.id2label)

        for k, v in metrics.items():
            self.log("test_"+k, v, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, lens, y = batch
        return self.forward(x, lens)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class RNNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, pad_idx, rnn_units, num_labels, id2label, num_layers=1,
                 bidirectional=False, dropout=0.25, embedding_matrix=None, freeze_embeddings=True):

        super().__init__(id2label)

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, padding_idx=pad_idx,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim,
                padding_idx = pad_idx)

        self.rnn = nn.GRU(embedding_dim,
                   rnn_units,
                   num_layers=num_layers,
                   bidirectional=bidirectional, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        factor = 2 if bidirectional else 1

        self.fc = nn.Linear(rnn_units * factor, num_labels)

    def forward(self, text, text_lens):
        #text = [batch_size, text len]
        #permuted = text.permute(1, 0)
        # permuted shape [batch_size, sent len]
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            # WTF no sé por qué hago esto de cpu
            embedded, text_lens.to("cpu"), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.rnn(packed_embedded)
        # hidden is the last state of the

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        output = self.dropout(output)
        # output is shape [seq, batch, hid]
        s = output.permute(1, 0, 2)
        # now [batch, seq, hid]
        mean = s.sum(dim=1) / text_lens.view(-1, 1)

        return self.fc(mean)

class FFNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, pad_idx, hidden_units, num_labels, id2label, dropout=0.25, embedding_matrix=None, freeze_embeddings=True):
        super().__init__(id2label)

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, padding_idx=pad_idx,
                freeze=freeze_embeddings
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim,
                padding_idx = pad_idx)

        self.encoder = nn.Linear(embedding_dim, hidden_units)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_units, num_labels)

    def forward(self, text, text_lens):
        #text = [batch_size, text len]
        #permuted = text.permute(1, 0)
        # permuted shape [batch_size, sent len]
        embedded = self.embedding(text)
        mean_embedding = embedded.mean(dim=1)

        encoded = F.relu(self.encoder(mean_embedding))
        encoded = self.dropout(encoded)
        return self.fc(encoded)
