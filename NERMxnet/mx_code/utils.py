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

# 一些通用函数
# author: kenjewu

import os
import pickle
import numpy as np
from gluonnlp.data import count_tokens
from gluonnlp import Vocab


def load_ne_cate_dic(path):
    '''
    加载命名实体的类别到符号标记的索引字典
    Args: 
        path: 索引字典存储的路径
    Returns:
        ne_cate_dic: 实体类别到标记的索引字典
    '''
    with open(path, 'rb') as fr:
        ne_cate_dic = pickle.load(fr)
    return ne_cate_dic


def read_data(word_path, label_path, nature_path, max_seq_len, PAD, NOT, PAD_NATURE, UNK):
    '''
    读取数据中的每个句子的词，词性，词所对应的实体的标记。对每条句子的词的长度进行长截短补到指定的
    max_seq_len 的长度，对词的填充使用 PAD, 词性填充使用 PAD_NATURE， 标记填充使用 NOT。
    构建 词的字典，词性的字典以及标记的字典，字典中保留位置符号 UNK
    Args:
        word_path: 包含每条句子的词的数据的路径
        label_path: 包含每条句子的词的标记的数据的路径
        nature_path: 包含每条句子的词的词性的数据的路径
        max_seq_len: 最大句子长度，以词为单位
        PAD: 词的填充符号
        NOT: 标记的填充符号
        PAD_NATURE: 词性的填充符号
        UNK: 未知符号
    Returns:
        word_vocab：词的字典 
        label_vocab：词所对应的实体的标记的字典
        nature_vocab：词的词性的字典
        input_seqs：所有句子的输入的词的列表 [[word1, word2, ...], [word1, word2, ...], ...]
        output_seqs: 所有句子的词的标记的列表 [[label1, label2, ...], [label1, label2, ...], ...]
        nature_seqs：所有句子的词的词性的列表 [[nature1, nature2, ...], [nature1, nature2, ...], ...]
    '''
    input_tokens, output_tokens, nature_tokens = [], [], []
    input_seqs, output_seqs, nature_seqs = [], [], []

    with open(word_path, 'r', encoding='utf-8') as fx, open(label_path, 'r', encoding='utf-8') as fy, open(nature_path, 'r', encoding='utf-8') as fn:
        word_lines = fx.readlines()
        label_lines = fy.readlines()
        word_natures = fn.readlines()
        assert len(word_lines) == len(word_natures)
        assert len(word_natures) == len(label_lines)

        for word_line, label_line, word_nature in zip(word_lines, label_lines, word_natures):
            input_seq = word_line.strip()
            output_seq = label_line.strip()
            nature_seq = word_nature.strip()

            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')
            cur_nature_tokens = nature_seq.split(' ')
            assert len(cur_input_tokens) == len(cur_output_tokens)
            assert len(cur_output_tokens) == len(cur_nature_tokens)

            # 跳过奇怪的实体类别标注
            if '' in cur_output_tokens:
                continue

            # if-else: 长截短补
            if len(cur_input_tokens) < max_seq_len or len(cur_output_tokens) < max_seq_len or len(cur_nature_tokens) < max_seq_len:

                # 添加 PAD 符号使每个序列长度都为 max_seq_len
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                    cur_output_tokens.append(NOT)
                    cur_nature_tokens.append(PAD_NATURE)
            else:
                cur_input_tokens = cur_input_tokens[0:max_seq_len]
                cur_output_tokens = cur_output_tokens[0:max_seq_len]
                cur_nature_tokens = cur_nature_tokens[0:max_seq_len]

            input_tokens.extend(cur_input_tokens)
            output_tokens.extend(cur_output_tokens)
            nature_tokens.extend(cur_nature_tokens)

            # 记录序列
            input_seqs.append(cur_input_tokens)
            output_seqs.append(cur_output_tokens)
            nature_seqs.append(cur_nature_tokens)

        # 创建字典
        word_vocab = Vocab(count_tokens(input_tokens), unknown_token=UNK, padding_token=PAD)
        label_vocab = Vocab(count_tokens(output_tokens), unknown_token=UNK, padding_token=NOT)
        nature_vocab = Vocab(count_tokens(nature_tokens), unknown_token=UNK, padding_token=PAD_NATURE)

    return word_vocab, label_vocab, nature_vocab, input_seqs, output_seqs, nature_seqs


def reverse_vocab(word_vocab, label_vocab, nature_vocab):
    '''
    构建索引到字符串的反转字典
    Args:
        word_vocab：词的字典 
        label_vocab：词所对应的实体的标记的字典
        nature_vocab：词的词性的字典
    Returns:
        word_id_to_word: 词索引到词的字典
        label_id_to_label： 标记索引到标记的字典
        nature_id_to_nature： 词性索引到词性的字典
    '''
    word_id_to_word = word_vocab.token_to_idx
    label_id_to_label = label_vocab.token_to_idx
    nature_id_to_nature = nature_vocab.token_to_idx
    return word_id_to_word, label_id_to_label, nature_id_to_nature


def convert_txt_data(X_path, y_path, nature_path,
                     input_seqs, output_seqs, nature_seqs,
                     word_vocab, label_vocab, nature_vocab, max_seq_len, UNK):
    '''
    将每个文本数据的字符串根据相应的索引字典数字化。如果已经数字化保存在了硬盘上，就直接读取。
    Args:
        X_path: 词的数字索引的数据的路径
        y_path：实体类别标签的数字索引的数据路径
        nature_path：词性的数字索引的数据的路径
        input_seqs：所有句子的输入的词的列表 [[word1, word2, ...], [word1, word2, ...], ...]
        output_seqs: 所有句子的词的标记的列表 [[label1, label2, ...], [label1, label2, ...], ...]
        nature_seqs：所有句子的词的词性的列表 [[nature1, nature2, ...], [nature1, nature2, ...], ...]
        word_vocab：词的字典
        label_vocab：实体类别标记的字典
        nature_vocab：词的词性的字典
        max_seq_len：最大句子长度，以词为单位
        UNK：未知符号
    Returns:
        data_x, data_y, data_nature 皆为 numpy 矩阵
    '''
    # 将文本数据数字化
    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(nature_path):
        print("loading...")
        data_x = np.load(X_path)
        data_y = np.load(y_path)
        data_nature = np.load(nature_path)
        print("end!")
    else:
        print("converting...")
        data_x = np.zeros((len(input_seqs), max_seq_len), dtype=np.int32)
        data_y = np.zeros_like(data_x, dtype=np.int32)
        data_nature = np.zeros_like(data_x, dtype=np.int32)

        for i in range(len(input_seqs)):
            data_x[i] = word_vocab.to_indices(input_seqs[i])
            data_y[i] = label_vocab.to_indices(output_seqs[i])
            data_nature[i] = nature_vocab.to_indices(nature_seqs[i])

        np.save(X_path, data_x)
        np.save(y_path, data_y)
        np.save(nature_path, data_nature)
        print("end!")
    return data_x, data_y, data_nature


def generate_named_entity(word_vocab, sentence_input, pred, max_seq_len, only_ne_cate_dic):
    '''
    通过模型预测的标记结果，将标记结构重新组合为实体字符串
    Args:
        word_vocab: 词典。 gluonnlp.Vocab 对象
        sentence_input: 一维，一条句子的词的整数索引的列表
        pred: 一维， 一条句子的实体类别标记的预测的列表
        max_seq_len: 最大句子长度，以词为单位
        only_ne_cate_dic: 只包含了是实体的实体类别的标记的字典
            {'组织机构名名实体': 'A', '地名实体': 'C', '人名实体': 'D'}
    Returns:
        ne_dict：存储每个实体类别的实体列表的字典
            {'组织机构名名实体': [], '地名实体': [], '人名实体': []}
    '''
    indices = []
    ne_dict = {key: [] for key in only_ne_cate_dic.values()}

    for word_idx in range(max_seq_len):
        pref = pred[word_idx][0]

        if pref in only_ne_cate_dic.values():
            endf = pred[word_idx][1]
            # 预测标记的结尾是 C，那肯定是个完成的实体
            if endf == 'C':
                ne_dict[pref].append(word_vocab.to_tokens(sentence_input[word_idx]))
                # ne_dict[pref].append(tl.nlp.word_ids_to_words([sentence_input[word_idx]], word_id_to_word)[0])
            # 预测标记的结尾是 B
            elif endf == 'B':
                # 没到词尾，但是满足实体了
                if word_idx+1 < len(pred):
                    # 如果后面那个词的标记不是双字母标记，后面那个词肯定不是实体
                    if len(pred[word_idx + 1]) != 2:
                        ne_dict[pref].append(word_vocab.to_tokens(sentence_input[word_idx]))
                        indices.clear()
                    elif (pred[word_idx + 1][1] != 'I' and pred[word_idx + 1][1] != 'E') or (pred[word_idx + 1][0] != pref):
                        ne_dict[pref].append(word_vocab.to_tokens(sentence_input[word_idx]))
                        indices.clear()
                # 到词尾，不管满不满足实体，都截断
                elif word_idx + 1 == len(pred):
                    ne_dict[pref].append(word_vocab.to_tokens(sentence_input[word_idx]))
                    indices.clear()
                else:
                    indices.append(word_idx)
            # 预测标记的结尾是 I
            elif endf == 'I':
                if word_idx + 1 < len(pred):
                    if len(pred[word_idx+1]) != 2:
                        concated_str = ''
                        for idx in indices:
                            concated_str += word_vocab.to_tokens(sentence_input[idx])
                        ne_dict[pref].append(concated_str)
                        indices.clear()
                    elif (pred[word_idx+1][1] != 'E' and pred[word_idx+1][1] != 'I') or (pred[word_idx+1][0] != pref):
                        concated_str = ''
                        for idx in indices:
                            concated_str += word_vocab.to_tokens(sentence_input[idx])
                        ne_dict[pref].append(concated_str)
                        indices.clear()
                elif word_idx+1 == len(pred):
                    concated_str = ''
                    for idx in indices:
                        concated_str += word_vocab.to_tokens(sentence_input[idx])
                    ne_dict[pref].append(concated_str)
                    indices.clear()
                else:
                    indices.append(word_idx)
            # 预测标记的结尾是 E
            elif endf == 'E':
                indices.append(word_idx)
                concated_str = ''
                for idx in indices:
                    concated_str += word_vocab.to_tokens(sentence_input[idx])
                ne_dict[pref].append(concated_str)
                indices.clear()
    return ne_dict


def generate_named_entity_with_offset(all_words, se_label_str, only_ne_cate_dic):
    '''
    生成实体字符串与实体的 offset
    :param all_words: 句子的所有的词
    :param se_label_str: 预测的实体的标记
    :param only_ne_cate_dic: 只包含了是实体的实体类别的标记的字典
            {'组织机构名名实体': 'A', '地名实体': 'C', '人名实体': 'D'}
    :return:
            pred_ne {offset: {实体字符串: 实体类别}}
    '''
    indices = []
    # {实体类别符号:实体类别字符串}
    cate_ne_dic = {value: key for key, value in only_ne_cate_dic.items()}
    pred_ne = {}
    offset = 0
    for word_num in range(len(se_label_str)):
        pref = se_label_str[word_num][0]
        if pref in only_ne_cate_dic.values():
            endf = se_label_str[word_num][1]

            if endf == 'C':
                offset = get_ne_b(cate_ne_dic, indices, offset, pred_ne, pref, all_words, word_num)
            if endf == 'B':
                if word_num + 1 < len(se_label_str):
                    if len(se_label_str[word_num+1]) != 2:
                        offset = get_ne_b(cate_ne_dic, indices, offset, pred_ne, pref, all_words, word_num)
                    else:
                        temp_pref = se_label_str[word_num+1][0]
                        temp_endf = se_label_str[word_num+1][1]
                        if (temp_endf != 'I' and temp_endf != 'E') or temp_pref != pref:
                            offset = get_ne_b(cate_ne_dic, indices, offset, pred_ne, pref, all_words, word_num)
                        else:
                            # 考虑下是否需要，晕
                            indices.append(word_num)
                else:
                    if word_num + 1 == len(se_label_str):
                        offset = get_ne_b(cate_ne_dic, indices, offset, pred_ne, pref, all_words, word_num)
                    else:
                        indices.append(word_num)

            if endf == 'I':
                if word_num + 1 < len(se_label_str):
                    if len(se_label_str[word_num+1]) != 2:
                        offset = get_ne_i(cate_ne_dic, indices, offset, pred_ne, pref, all_words)
                    else:
                        temp_pref = se_label_str[word_num + 1][0]
                        temp_endf = se_label_str[word_num + 1][1]
                        if (temp_endf != 'I' and temp_endf != 'E') or temp_pref != pref:
                            offset = get_ne_i(cate_ne_dic, indices, offset, pred_ne, pref, all_words)
                        else:
                            # 考虑下是否需要，晕
                            indices.append(word_num)
                else:
                    if word_num + 1 == len(se_label_str):
                        offset = get_ne_i(cate_ne_dic, indices, offset, pred_ne, pref, all_words)
                    else:
                        indices.append(word_num)
            if endf == 'E':
                indices.append(word_num)
                offset = get_ne_i(cate_ne_dic, indices, offset, pred_ne, pref, all_words)

        else:
            offset += len(all_words[word_num])

    return pred_ne


def get_ne_i(cate_ne_dic, indices, offset, pred_ne, pref, all_words):
    '''
    generate_named_entity_with_offset 的辅助函数
    '''
    ne = ''
    for idx in indices:
        ne += all_words[idx]
    ne_and_cate = {ne: cate_ne_dic[pref]}
    pred_ne[offset] = ne_and_cate
    offset += len(ne)
    indices.clear()
    return offset


def get_ne_b(cate_ne_dic, indices, offset, pred_ne, pref, all_words, word_num):
    '''
    generate_named_entity_with_offset 的辅助函数
    '''
    ne = all_words[word_num]
    ne_and_cate = {ne: cate_ne_dic[pref]}
    pred_ne[offset] = ne_and_cate
    offset += len(ne)
    indices.clear()

    return offset


def cal_prf1(y_pred, y_true, sentences_input, label_vocab, word_vocab, max_seq_len, only_ne_cate_dic):
    '''
    计算实体预测结果的 Precision, Recall, F1
    Args:
        y_pred: 每条句子的每个词的实体标签数字索引的预测结果
            [[label1, label2, ...], [label1, label2, ...], ...]
        y_true: 每条句子的每个词的实体标签数字索引的真实结果, 形状和 y_pred 一样
        sentences_inputs: 每条句子的词的整数索引的列表
        label_vocab: 实体类别索引到实体类别字符串的字典
        word_vocab: 词索引到词字符串的字典
        max_seq_len: 最大句子长度，以词为单位
        only_ne_cate_dic: 只包含了是实体的实体类别的标记的字典
            {'组织机构名名实体': 'A', '地名实体': 'C', '人名实体': 'D'}

    Returns:
        prf_dic: 存储每个类别的实体的 P, R, F1值的字典
            {A：[p, r, f1], C: [p, r, f1], D: [p, r, f1]}
    '''

    # 根据真实标签和句子输入检索出真实的命名实体列表
    true_named_entity_dic = {key: [] for key in only_ne_cate_dic.values()}
    for idx, label_id in enumerate(y_true):
        sentence_input = sentences_input[idx]
        # 实体类别标记索引转实体类别的字符串
        true_label = label_vocab.to_tokens(label_id)
        # 生成实体字符串
        one_named_entity_dic = generate_named_entity(word_vocab, sentence_input,
                                                     true_label, max_seq_len, only_ne_cate_dic)
        ne_cates = true_named_entity_dic.keys()
        for key in ne_cates:
            # extend 是原地操作，返回 None
            true_named_entity_dic[key].extend(one_named_entity_dic[key])

    # 根据预测标签和句子输入检索出预测的命名实体列表
    pred_named_entity_dic = {key: [] for key in only_ne_cate_dic.values()}
    for idx, label_id in enumerate(y_pred):
        sentence_input = sentences_input[idx]
        pred_label = label_vocab.to_tokens(label_id)
        one_pred_named_entity_dic = generate_named_entity(word_vocab, sentence_input,
                                                          pred_label, max_seq_len, only_ne_cate_dic)
        ne_cates = pred_named_entity_dic.keys()
        for key in ne_cates:
            pred_named_entity_dic[key].extend(one_pred_named_entity_dic[key])

    # 计算 P R F1
    prf_dic = {key: [] for key in only_ne_cate_dic.values()}
    for key in prf_dic.keys():

        pred_named_entity_ls = pred_named_entity_dic[key]
        true_named_entity_ls = true_named_entity_dic[key]
        TP, FP, FN = 0.0, 0.0, 0.0
        for pred_ne, true_ne in zip(pred_named_entity_ls, true_named_entity_ls):
            if pred_ne in true_named_entity_ls:
                TP += 1
            if pred_ne not in true_named_entity_ls:
                FP += 1
            if true_ne not in pred_named_entity_ls:
                FN += 1
        if TP + FP == 0 or TP+FN == 0 or TP == 0:
            P, R, F1 = 0.0, 0.0, 0.0
        else:
            P = TP / (TP+FP)
            R = TP / (TP+FN)
            F1 = 2 * P * R / (P + R)
        prf_dic[key] = [P, R, F1]

    return prf_dic


def convert_signal_to_ne_name(only_ne_cate_dic, raw_prf_dic):
    '''
    将 {A：[p, r, f1], C: [p, r, f1], D: [p, r, f1]}
    转为：{'组织机构名名实体': [p, r, f1], '地名实体': [p, r, f1], '人名实体': [p, r, f1]}
    Args:
        only_ne_cate_dic: 只包含了是实体的实体类别的标记的字典
            {'组织机构名名实体': 'A', '地名实体': 'C', '人名实体': 'D'}
        raw_prf_dic: 原始计算得到的实体类别的 p r f1 的字典
            {A：[p, r, f1], C: [p, r, f1], D: [p, r, f1]}
    Returns:
        new_prf_dic: {'组织机构名名实体': [p, r, f1], '地名实体': [p, r, f1], '人名实体': [p, r, f1]}
    '''
    signal_to_ne_dic = {value: key for key, value in only_ne_cate_dic.items()}
    new_prf_dic = {}
    for key, value in raw_prf_dic.items():
        new_prf_dic[signal_to_ne_dic[key]] = value
    return new_prf_dic


def write_dic_to_txt(dic, path):
    '''
    将字典写到指定路径下
    Args:
        dic:字典
        path:路径
    '''
    print('writting dic to: ' + path)
    with open(path, 'w', encoding='utf-8') as fw:
        for idx, (key, value) in enumerate(dic.items()):
            if idx == len(dic) - 1:
                content = str(key)+':'+str(value)
            else:
                content = str(key)+':'+str(value)+'\n'
            fw.write(content)
    print('write end!')
