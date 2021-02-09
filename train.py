import gensim
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, early_stopping

from preprocessor import Preprocessor
from dataset import Dataset
from net import CNN


def get_dataloader(file_path, max_len, preprocessor, batch_size):
    dataset = Dataset(file_path, max_len, preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def main(config):
    word2vec_model = gensim.models.Word2Vec.load("data/ko.bin")
    word2vec_model.wv["<pad>"] = np.zeros(word2vec_model.wv.vector_size)
    word2vec_model.wv["<unk>"] = np.zeros(word2vec_model.wv.vector_size)

    preprocessor = Preprocessor(word2vec_model)
    train_dataloader = get_dataloader(
        "data/ratings_train.txt", config.max_len, preprocessor, config.batch_size
    )
    val_dataloader = get_dataloader(
        "data/ratings_test.txt", config.max_len, preprocessor, config.batch_size
    )

    logger = TensorBoardLogger(config.log_dir, config.cnn_type, config.task)
    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoint/{config.cnn_type}/{config.task}",
        filename="cnn-{epoch:02d}-{val_loss:.5f}",
        save_top_k=-1,
    )
    early_stopping = EarlyStopping("val_loss")

    net = CNN(word2vec_model.wv, config)
    trainer = pl.Trainer(
        distributed_backend="ddp",
        gpus=8,
        max_epochs=20,
        logger=logger,
        callbacks=[model_checkpoint, early_stopping],
    )
    trainer.fit(net, train_dataloader, val_dataloader)


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)
