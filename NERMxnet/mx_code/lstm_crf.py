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

# 一个用 LSTM 与 CRF 配合的神经网络进行命名实体识别的模型
# author: kenjewu

import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.nn import Dense, Embedding

from crf import CRF

warnings.filterwarnings('ignore')


class LSTM_CRF(nn.Block):
    def __init__(self, word_vocab_size, word_vec_size, nature_vocab_size, nature_vec_size,
                 hidden_dim, num_layers, tag2idx, drop_prob,  **kwargs):
        super(LSTM_CRF, self).__init__(**kwargs)
        with self.name_scope():
            self.drop_prob = drop_prob
            self.word_embdding = Embedding(word_vocab_size, word_vec_size)
            self.nature_embedding = Embedding(nature_vocab_size, nature_vec_size)
            self.lstm_layer = rnn.LSTM(hidden_dim, num_layers=num_layers, dropout=drop_prob, bidirectional=True)
            self.dense = Dense(len(tag2idx))
            self.crf = CRF(tag2idx)

    def forward(self, x, nature, state):
        x_embed = self.word_embdding(x)
        nature_embed = self.nature_embedding(nature)
        # lstm_input format: (T, N, C)
        lstm_input = nd.transpose(nd.concat(x_embed, nature_embed, dim=-1), (1, 0, 2))
        # lstm_output format: (T, N, C)
        lstm_output = self.lstm_layer(lstm_input)
        t, n, c = lstm_output.shape
        dense_input = nd.reshape(lstm_output, shape=(-1, c))
        feats = self.dense(dense_input).reshape((t, n, -1))
        score, tag_seq = self.crf(feats)
        return score, tag_seq, feats, state

    def begin_state(self, *args, **kwargs):
        return self.lstm_layer.begin_state(*args, **kwargs)
