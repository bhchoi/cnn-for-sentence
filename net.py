import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import accuracy_score


class CNN(pl.LightningModule):
    def __init__(self, word_vector, config):
        super().__init__()

        self.config = config
        self.word_vector = word_vector
        self.emb_dim = self.word_vector.vector_size

        if self.config.cnn_type == "CNN-rand":
            self.embedding = nn.Embedding(len(self.word_vector.vocab), self.emb_dim)

        elif self.config.cnn_type == "CNN-static":
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.word_vector.vectors),
                freeze=True,
                padding_idx=self.word_vector.vocab["<pad>"].index,
            )

        elif self.config.cnn_type == "CNN-none-static":
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.word_vector.vectors),
                freeze=False,
                padding_idx=self.word_vector.vocab["<pad>"].index,
            )

        elif self.config.cnn_type == "CNN-multichannel":
            self.embedding_static = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.word_vector.vectors),
                freeze=True,
                padding_idx=self.word_vector.vocab["<pad>"].index,
            )

            self.embedding_non_static = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.word_vector.vectors),
                freeze=False,
                padding_idx=self.word_vector.vocab["<pad>"].index,
            )

        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.emb_dim,
                    out_channels=self.config.filter_size,
                    kernel_size=ks,
                )
                for ks in self.config.kernel_sizes
            ]
        )

        self.pool = nn.MaxPool1d(kernel_size=self.config.max_pooling_kernel_size)
        fc_size = np.sum(
            [
                int(
                    (self.config.max_len - ks + 1) / self.config.max_pooling_kernel_size
                )
                * self.config.filter_size
                for ks in self.config.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(fc_size, 2)

    def forward(self, input_ids):
        if self.config.cnn_type == "CNN-multichannel":
            x_static = self.embedding_static(input_ids)
            x_non_static = self.embedding_non_static(input_ids)

            x_static = x_static.permute(0, 2, 1)
            x_non_static = x_non_static.permute(0, 2, 1)

            x_conv_list = [
                self.pool(F.relu(conv(x_static)))
                + self.pool(F.relu(conv(x_non_static)))
                for conv in self.conv_list
            ]

        else:
            x = self.embedding(input_ids)
            x = x.permute(0, 2, 1)
            x_conv_list = [self.pool(F.relu(conv(x))) for conv in self.conv_list]

        x_fc = torch.cat(
            [x_pool.view(x_pool.size()[0], -1) for x_pool in x_conv_list], dim=1
        )

        x = self.fc(x_fc)
        x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_nb):
        input_ids, labels = batch
        output = self(input_ids)
        loss = F.cross_entropy(output, labels)
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        input_ids, labels = batch
        output = self(input_ids)
        output = output.squeeze(1)
        val_loss = nn.CrossEntropyLoss()(output, labels)
        prediction = output.max(1)[1]
        val_acc = torch.tensor(
            accuracy_score(labels.cpu().numpy(), prediction.cpu().numpy()),
            dtype=torch.float32,
        )

        return {"val_loss": val_loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val_acc", val_acc, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch, batch_nb):
        input_ids, labels = batch
        output = self(input_ids)
        output = output.squeeze(1)
        prediction = output.max(1)[1]
        test_acc = torch.tensor(
            accuracy_score(labels.cpu().numpy(), prediction.cpu().numpy()),
            dtype=torch.float32,
        )

        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("test_acc", test_acc, prog_bar=True, logger=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
