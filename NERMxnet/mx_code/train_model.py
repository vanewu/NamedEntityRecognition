# Copyright https://github.com/kenjewu/NamedEntityRecognize kenjewu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 训练模型
#   python train_model.py -mn xx -mp xx
#   -mn: 要训练的模型的名字    -mp: 模型训练好后保存的位置，默认：'../models/mx/lstm_crf_model.params'
# author: kenjewu

import argparse
import os
import pickle

import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon.data import ArrayDataset, DataLoader
from sklearn.model_selection import train_test_split

import train_helper as th
from config import Configuer
from lstm_crf import LSTM_CRF
from utils import *

CTX = mx.gpu()


def main(model_saved_path, model_name):
    ne_cate_dic = Configuer.ne_cate_dic
    word_path = Configuer.word_path
    label_path = Configuer.label_path
    nature_path = Configuer.nature_path

    X_path = Configuer.X_path
    y_path = Configuer.y_path
    nature_py_path = Configuer.nature_py_path
    word_vocab_path = Configuer.word_vocab_path
    label_vocab_path = Configuer.label_vocab_path
    nature_vocab_path = Configuer.nature_vocab_path

    max_seq_len = Configuer.MAX_SEQ_LEN
    pad = Configuer.PAD
    pad_nature = Configuer.PAD_NATURE
    unk = Configuer.UNK
    not_ne = Configuer.NOT

    # 从本地读取数据
    if os.path.exists(word_vocab_path) and os.path.exists(label_vocab_path)\
            and os.path.exists(nature_vocab_path) and os.path.exists(X_path)\
            and os.path.exists(y_path) and os.path.exists(nature_py_path):
        print('Loading existed data...')
        with open(word_vocab_path, 'rb') as f1, open(label_vocab_path, 'rb') as f2, open(nature_vocab_path, 'rb') as f3:
            word_vocab = pickle.load(f1)
            label_vocab = pickle.load(f2)
            nature_vocab = pickle.load(f3)
        data_x, data_y, data_nature = np.load(X_path), np.load(y_path), np.load(nature_py_path)
        print('Loading end!')
    else:
        # 转换文本数据到 numpy数据 和 pickle 数据
        print('Converting data from scratch...')
        word_vocab, label_vocab, nature_vocab, input_seqs, output_seqs, nature_seqs = read_data(
            word_path, label_path, nature_path, max_seq_len, pad, not_ne, pad_nature, unk)
        data_x, data_y, data_nature = convert_txt_data(X_path, y_path, nature_py_path,
                                                       input_seqs, output_seqs, nature_seqs,
                                                       word_vocab, label_vocab, nature_vocab, max_seq_len, unk)
        with open(word_vocab_path, 'wb') as fw1, open(label_vocab_path, 'wb') as fw2, open(nature_vocab_path, 'wb') as fw3:
            pickle.dump(word_vocab, fw1)
            pickle.dump(label_vocab, fw2)
            pickle.dump(nature_vocab, fw3)
        np.save(X_path, data_x)
        np.save(y_path, data_y)
        np.save(nature_py_path, data_nature)
        print('Converting end!')

    # 切分训练集和验证集
    X_train, X_valid, Y_train, Y_valid, nature_train, nature_valid = train_test_split(
        data_x, data_y, data_nature, test_size=0.1, random_state=33)
    print(X_train.shape, X_valid.shape)
    # X_train = X_train[0:512]
    # nature_train = nature_train[0:512]
    # Y_train = Y_train[0:512]
    # X_valid = X_valid[0:512]
    # nature_valid = nature_valid[0:512]
    # Y_valid = Y_valid[0:512]
    dataset_train = ArrayDataset(
        nd.array(X_train, ctx=CTX),
        nd.array(nature_train, ctx=CTX),
        nd.array(Y_train, ctx=CTX))
    data_iter_train = DataLoader(dataset_train, batch_size=256, shuffle=True, last_batch='rollover')
    dataset_valid = ArrayDataset(
        nd.array(X_valid, ctx=CTX),
        nd.array(nature_valid, ctx=CTX),
        nd.array(Y_valid, ctx=CTX))
    data_iter_valid = DataLoader(dataset_valid, batch_size=256, shuffle=False)

    # 根据参数配置模型
    model, loss = None, None
    word_vocab_size, word_vec_size = len(word_vocab), 300
    nature_vocab_size, nature_vec_size = len(nature_vocab), 50
    drop_prob = 0.3
    num_epochs = 20
    lr = 0.0001

    if model_name == 'lstm_crf':
        print('train lstm_crf model')
        hidden_dim = 128
        num_layers = 2
        tag2idx = label_vocab.token_to_idx
        model = LSTM_CRF(word_vocab_size, word_vec_size, nature_vocab_size, nature_vec_size,
                         hidden_dim, num_layers, tag2idx, drop_prob)
        model.initialize(init=init.Xavier(), ctx=CTX)
        loss = model.crf.neg_log_likelihood
    elif model_name == 'cnn_crf':
        pass
    elif model_name == 'cnn':
        pass

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})

    # 开始训练
    print('waiting...')
    print(model)
    th.train(data_iter_train, data_iter_valid, model, loss, trainer, CTX, num_epochs,
             word_vocab, label_vocab, max_seq_len, ne_cate_dic)

    # 保存模型参数
    model.save_parameters(model_saved_path)
    print(model_name + 'model params has saved in :', os.path.abspath(model_saved_path))


if __name__ == '__main__':
    parse = argparse.ArgumentParser('some params')
    parse.add_argument('-mp', dest='model_saved_path', action='store', default='../models/mx/lstm_crf_model.params')
    parse.add_argument('-mn', dest='model_name', action='store', required=1)
    args = parse.parse_args()
    main(args.model_saved_path, args.model_name)
