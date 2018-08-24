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

# 训练模型时的一些辅助通用函数
# author: kenjewu

import sys
from time import time

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd, gluon, nd

from utils import cal_prf1, convert_signal_to_ne_name

sys.path.append('..')


def train(data_iter_train, data_iter_valid, model, loss, trainer, CTX, num_epochs,
          word_vocab, label_vocab, max_seq_len, ne_cate_dic):
    print('Train on ', CTX)
    only_ne_cate_dic = ne_cate_dic.copy()
    only_ne_cate_dic.pop('不是实体')
    print(only_ne_cate_dic)
    print(ne_cate_dic)
    for epoch in range(1, num_epochs + 1):
        start = time()
        states = None
        for n_batch, (batch_x, batch_nature, batch_y) in enumerate(data_iter_train):
            with autograd.record():
                batch_score, batch_pred, feats, _ = model(batch_x, batch_nature, states)
                l = loss(feats, nd.split(batch_y, max_seq_len, axis=1))
            l.backward()
            trainer.step(batch_x.shape[0])

            # 每隔 skip_step ,采样看看
            if (n_batch+1) % 100 == 0:
                print("Epoch {0}, n_batch {1}, loss {2}".format(epoch, n_batch+1, l.mean().asscalar()))
                batch_y = batch_y.asnumpy().astype(np.int32, copy=False)
                batch_pred = batch_pred.asnumpy().astype(np.int32, copy=False)
                for example in range(3):
                    true_idx = batch_y[example].tolist()
                    pred_idx = batch_pred[example].tolist()

                    true_label = label_vocab.to_tokens(true_idx)
                    pred_label = label_vocab.to_tokens(pred_idx)

                    print("    Sample {0}: ".format(example))
                    print("    True Label {0}: ".format(true_label))
                    print("    Pred Label {0}: ".format(pred_label))
        # 在训练集上评估
        print('Evaluating...')

        prf_dic_train, train_loss = evaluate(data_iter_train, model, states, loss,
                                             word_vocab, label_vocab, max_seq_len, only_ne_cate_dic)
        prf_dic_valid, valid_loss = evaluate(data_iter_valid, model, states, loss,
                                             word_vocab, label_vocab, max_seq_len, only_ne_cate_dic)

        print("===========================================")
        print("Epoch {0}, epoch_loss_train {1}, epoch_loss_valid {2}".format(epoch, train_loss, valid_loss))
        print(prf_dic_train)
        print(prf_dic_valid)
        print("===========================================")
        print()


def evaluate(data_iter_valid, model, state, loss, word_vocab, label_vocab, max_seq_len, only_ne_cate_dic):
    valid_loss = 0.

    y_true, y_pred, sentences_input = [], [], []
    for n_batch, (batch_x, batch_nature, batch_y) in enumerate(data_iter_valid):
        batch_score, batch_pred, feats, _ = model(batch_x, batch_nature, state)
        l = loss(feats, nd.split(batch_y, max_seq_len, axis=1))

        y_pred.append(batch_pred.asnumpy().astype(np.int32, copy=False))
        y_true.append(batch_y.asnumpy().astype(np.int32, copy=False))
        sentences_input.append(batch_x.asnumpy().astype(np.int32, copy=False))

        valid_loss += l.mean().asscalar()

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    sentences_input = np.vstack(sentences_input)
    valid_loss /= (n_batch + 1)

    # 计算训练集上的 P R F1
    raw_prf_dic = cal_prf1(y_pred.tolist(), y_true.tolist(),
                           sentences_input.tolist(), label_vocab,
                           word_vocab, max_seq_len, only_ne_cate_dic
                           )

    prf_dic = convert_signal_to_ne_name(only_ne_cate_dic, raw_prf_dic)
    prf_dic = pd.DataFrame(list(prf_dic.values()), index=list(prf_dic.keys()),
                           columns=['P', 'R', 'F1'])

    return prf_dic, valid_loss
