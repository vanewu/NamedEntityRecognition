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

# 使用已经训练并保存好后的模型预测，作为服务器端运行，使用 Flask 框架
# pyton predict.py -mn xx -mp xx
#       -mn 模型名字,默认 lstm_crf    -mp 模型参数的位置,默认 '../models/mx/lstm_crf_model.params'
# author: kenjewu


import argparse
import pickle
import sys

import flask
import jieba
import mxnet as mx
import numpy as np
from flask import Flask, request
from mxnet import nd

from config import Configuer
from lstm_crf import LSTM_CRF
from utils import generate_named_entity_with_offset

app = Flask(__name__)
CTX = mx.gpu()


def get_vocabs():
    '''
    获取四个词典
    word_vocab：词的词典
    nature_vocab：词性的词典
    label_vocab：标记的词典
    nature_search_dict：词的词性的搜索字典
    '''
    try:
        with open(Configuer.word_vocab_path, 'rb') as f1, open(Configuer.label_vocab_path, 'rb') as f2,\
                open(Configuer.nature_vocab_path, 'rb') as f3, open(Configuer.nature_search_dict_path, 'rb') as f4:
            global word_vocab, label_vocab, nature_vocab, nature_search_dict
            word_vocab = pickle.load(f1)
            label_vocab = pickle.load(f2)
            nature_vocab = pickle.load(f3)
            nature_search_dict = pickle.load(f4)
        # return word_vocab, label_vocab, nature_vocab, nature_search_dict
    except IOError as ioe:
        raise ioe


def find_nature(words, nature_search_dict):
    '''
    查找词的词性
    Args:
        words:词的字符列表
        nature_search_dict: 词的词性的索引字典
    Returns:
        natures: 词性的列表
    '''
    natures = []
    for word in words:
        if word in nature_search_dict.keys():
            natures.append(nature_search_dict[word])
        elif word == Configuer.PAD:
            natures.append(Configuer.PAD_NATURE)
        else:
            natures.append(Configuer.UNK)
    return natures


def get_inputs(sentence, word_vocab, nature_search_dict, nature_vocab, label_vocab):
    '''
    对句子进行一系列的处理，使其符合模型的输入
    Args:
        sentence: 句子字符串
        word_vocab：词的词典
        nature_vocab：词性的词典
        label_vocab：标记的词典
        nature_search_dict：词的词性的搜索字典
    Returns:
        all_words: 包含 PAD 的所有词
        all_natures: 包含 PAD 的所有词性
        all_words_idx: 包含 PAD 的所有词的索引
        all_natures_idx: 包含 PAD 的所有词性的索引
        sentence_length: 原始字符串的长度
    '''
    max_seq_len = Configuer.MAX_SEQ_LEN
    cuted_sentence = list(jieba.cut(sentence))
    sentence_length = len(cuted_sentence)

    all_words = []
    if sentence_length < max_seq_len:
        all_words.append(cuted_sentence + [Configuer.PAD] * (max_seq_len - sentence_length))
    elif sentence_length > max_seq_len:
        for start in range(0, sentence_length, max_seq_len):
            if start + max_seq_len > sentence_length:
                sub_words = cuted_sentence[start:sentence_length] + [Configuer.PAD] * \
                    (max_seq_len - len(cuted_sentence[start: sentence_length]))
            else:
                sub_words = cuted_sentence[start:start+max_seq_len]
            all_words.append(sub_words)
    else:
        all_words.append(cuted_sentence)

    all_words_idx = []
    all_natures_idx = []
    all_natures = []
    for words in all_words:
        words_idx = word_vocab.to_indices(words)
        all_words_idx.append(words_idx)
        natures = find_nature(words, nature_search_dict)
        all_natures.append(natures)
        natures_idx = nature_vocab.to_indices(natures)
        all_natures_idx.append(natures_idx)

    return all_words, all_natures, all_words_idx, all_natures_idx, sentence_length


def load_model(model_name, model_saved_path, word_vocab, nature_vocab, label_vocab):
    '''
    加载模型
    Args:
        model_name: 模型的名字
        model_saved_path：模型参数保存的位置
        word_vocab：词的词典
        nature_vocab：词性的词典
        label_vocab：标记的词典
    '''
    word_vocab_size, word_vec_size = len(word_vocab), 300
    nature_vocab_size, nature_vec_size = len(nature_vocab), 50
    drop_prob = 0.3

    global model
    if model_name == 'lstm_crf':
        hidden_dim = 128
        num_layers = 2
        tag2idx = label_vocab.token_to_idx
        model = LSTM_CRF(word_vocab_size, word_vec_size, nature_vocab_size, nature_vec_size,
                         hidden_dim, num_layers, tag2idx, drop_prob)
        model.load_parameters(model_saved_path, ctx=CTX)
    elif model_name == 'cnn_crf':
        pass
    elif model_name == 'cnn':
        pass

    # return model


def predict(model, model_name, words_idx, nature_idx, all_words, sentence_length, label_vocab):
    '''
    预测函数
    Args:
        model: 模型
        model_name: 模型名字
        words_idx : 词的索引输入
        nature_idx : 词性的索引输入
        all_words: 所有的词，包括 PAD
        sentence_length: 原始句子的长度，未 PAD的
        label_vocab: 标记的词典
    Return:
        pred_ne：预测的实体的信息 {offset:{实体字符串：实体类别}}
    '''
    ne_cate_dic = Configuer.ne_cate_dic
    only_ne_cate_dic = ne_cate_dic.copy()
    only_ne_cate_dic.pop('不是实体')

    words_inputs, nature_inputs = nd.array(words_idx, ctx=CTX), nd.array(nature_idx, ctx=CTX)
    if model_name == 'lstm_crf':
        state = None
        score, pred, feats, _ = model(words_inputs, nature_inputs, state)

    # 预测结果进行解析
    temp = []
    for x in pred.asnumpy().astype(np.int32).tolist():
        temp.extend(x)
    result = temp[0:sentence_length]
    se_label_str = label_vocab.to_tokens(result)

    # 获取原始字符
    sentence_words = []
    for words in all_words:
        sentence_words.extend(words)
    sentence_words = sentence_words[0:sentence_length]

    # 生成实体字符串和其 offset
    # pred_ne {offset:{实体字符串：实体类别}}
    pred_ne = generate_named_entity_with_offset(sentence_words, se_label_str, only_ne_cate_dic)
    return pred_ne


@app.route('/ss', methods=['POST', 'GET'])
def launch():
    '''
    根据url 运行的函数，用于获取传入的句子字符串，并进行一系列处理，最后放入模型预测
    '''
    data = {}
    if request.method == 'POST':
        assert word_vocab is not None
        assert nature_vocab is not None
        assert label_vocab is not None
        assert nature_search_dict is not None
        sentence = request.form['sentence']
        all_words, all_natures, all_words_idx, all_natures_idx, sentence_length = get_inputs(
            sentence, word_vocab, nature_search_dict, nature_vocab, label_vocab)
        pred_ne = predict(model, model_name, all_words_idx, all_natures_idx, all_words,
                          sentence_length, label_vocab)

        data['sentence'] = sentence
        data['ner'] = pred_ne
        return flask.jsonify(data)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='解析预测参数')
    parse.add_argument('-mn', dest='model_name', action='store', default='lstm_crf')
    parse.add_argument('-mp', dest='model_saved_path', action='store', default='../models/mx/lstm_crf_model.params')

    # 解析所要使用的模型名字和对应的模型保存的位置参数
    args = parse.parse_args()
    global model_name
    model_name = args.model_name
    model_saved_path = args.model_saved_path

    # 获取指定的几个词典
    get_vocabs()
    # 加载模型
    load_model(model_name, model_saved_path, word_vocab, nature_vocab, label_vocab)
    print(model)

    # 运行
    app.run(debug=True)
