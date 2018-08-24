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

# 模型的一些固定参数的配置模块
# author: kenjewu

from utils import load_ne_cate_dic


class Configuer():
    ne_cate_dic_path = '../data/ne_cate_dic.pkl'
    ne_cate_dic = load_ne_cate_dic(ne_cate_dic_path)
    word_path = '../data/sentences_word.txt'
    label_path = '../data/sentences_label.txt'
    nature_path = '../data/sentences_nature.txt'

    X_path = '../data/numpy_data/X.npy'
    y_path = '../data/numpy_data/y.npy'
    nature_py_path = '../data/numpy_data/nature.npy'
    word_vocab_path = '../data/vocabs/gen_word_vocab.pkl'
    label_vocab_path = '../data/vocabs/label_vocab.pkl'
    nature_vocab_path = '../data/vocabs/nature_vocab.pkl'
    nature_search_dict_path = '../data/vocabs/nature_search_dict.pkl'

    MAX_SEQ_LEN = 30
    PAD = '<PAD>'
    PAD_NATURE = 'r'
    UNK = '<UNK>'
    NOT = ne_cate_dic['不是实体']
