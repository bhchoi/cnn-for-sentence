# cnn-for-sentence
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) 논문에 대한 구현체

[논문 리뷰 바로가기](https://bhchoi.github.io/posts/Convolutional_neural_networks_for_sentence_classification)

## Requirements
* pytorch
* pytorch-lightning
* omegaconf
* gensim
* numpy
* sklearn

## Dataset
Naver sentiment movie corpus v1.0

## Word Vector
한국어 word2vec  
[https://github.com/Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors)

## Train
config/train_config.yaml
```yaml
task: "20210208"
batch_size: 256
dropout_rate: 0.5
max_len: 70
filter_size: 100
kernel_sizes: [3, 4, 5]
max_pooling_kernel_size: 2
cnn_type: CNN-multichannel
log_dir: logs
```
```shell
python train.py
```
## Eval
```yaml
ckpt_path: checkpoint/CNN-multichannel/20210208/cnn-epoch=04-val_loss=0.49028.ckpt
batch_size: 256
dropout_rate: 0.5
max_len: 70
filter_size: 100
kernel_sizes: [3, 4, 5]
max_pooling_kernel_size: 2
cnn_type: CNN-multichannel
```
```shell
python eval.py
```

## Reference
[https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)  
[https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)  
[https://github.com/Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors)
