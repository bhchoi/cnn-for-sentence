import torch
import pytorch_lightning as pl
import gensim
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import Dataset
from net import CNN


def get_dataloader(file_path, max_len, preprocessor, batch_size):
    dataset = Dataset(file_path, max_len, preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader 


def main(config):
    word2vec_model = gensim.models.Word2Vec.load(config.pretrained_word_vector)
    word2vec_model.wv["<pad>"] = np.zeros(word2vec_model.wv.vector_size)
    word2vec_model.wv["<unk>"] = np.zeros(word2vec_model.wv.vector_size)

    preprocessor = Preprocessor(word2vec_model)

    test_dataloader = get_dataloader(
        config.test_data, config.max_len, preprocessor, config.batch_size
    )

    net = CNN(word2vec_model.wv, config)
    checkpoint = torch.load(config.ckpt_path)
    net.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer(
        distributed_backend=config.distributed_backend,
        gpus=config.gpus,
    )
    res = trainer.test(net, test_dataloader)


if __name__ == "__main__":
    config = OmegaConf.load("config/eval_config.yaml")
    main(config)