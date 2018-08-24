[![GitHub license](https://img.shields.io/badge/license-Apache2.0-blue.svg)](./LICENSE)

# Named Entity Recognition
***
## Instruction
***
This repository contains various experimental methods for deep-learning-based Chinese named entity recognition in natural language processing.
## Environment
***
Experimental development environment: python3, numpy, mxnet, sklearn, matplotlib, pandas.

## Algorithm List
***
[![CNN Model](https://img.shields.io/badge/Model-CNN-brightgreen.svg) ]() &nbsp;&nbsp;Convolutional Neural Network for Named Entity Recognition

[![LSTM_CRF Model](https://img.shields.io/badge/Model-LSTM__CRF-blue.svg) ](./mx_code/lstm_crf.py) &nbsp;&nbsp;LSTM + CRF  for Named Entity Recognition

[![CNN_CRF Model](https://img.shields.io/badge/Model-CNN__CRF-red.svg) ]() &nbsp;&nbsp;CNN + CRF  for Named Entity Recognition

[![CNN_LSTM_CRF Model](https://img.shields.io/badge/Model-CNN__LSTM__CRF-ff69b4.svg) ]() &nbsp;&nbsp;CNN + LSTM + CRF  for Named Entity Recognition

## Usage
***
1. ### Training model
        cd mx_code

        python train_model.py -mn xx -mp xx

        -mn: 要训练的模型的名字

        -mp: 模型训练好后保存的位置，默认：'../models/mx/lstm_crf_model.params'
2. ### Prediction
   After the training is completed and the model parameters are saved, there are two ways to predict.

    (1). The first is directly as a script application
        cd mx_code

        pyton predict.py sentence -mn xx -mp xx

        sentence 要预测的句子

        -mn 模型名字,默认 lstm_crf
        
        -mp 模型参数的位置,默认 '../models/mx/lstm_crf_model.params'

        example: python predict.py 计算机科学会议在四川省电子科技大学展开，大数据中心主任周涛介绍了目前人工智能的发展。
            result: {8: {'四川省电子科技大学': '组织机构名名实体'}, 27: {'周涛': '人名实体'}}

    (2). The second is as a Flask service application.
        cd mx_code
        
        pyton predict.py -mn xx -mp xx

        -mn 模型名字,默认 lstm_crf

        -mp 模型参数的位置,默认 '../models/mx/lstm_crf_model.params'

    After the model server is enabled, the client can access it through http://127.0.0.1:5000/ss

## Content
```
NERMxnet
│   README.md  介绍文件  
│
└───data
│   │
│   │───numpy_data              numpy数据文件夹
│   │───vocabs                  训练词典保存文件夹
│   │───ne_cate_dic.pkl         实体类别编号字典
│   │───sentences_label.txt     训练数据的标记
│   │───sentences_nature.txt    训练数据的词性
│   │───sentences_word.txt      训练数据的词
│   
└───models
│   │   
│   │───mx                      模型参数保存文件夹
│   │    │
│   
└───mx_code
│   │   
│   │───config.py               配置文件
│   │   
│   │───crf.py                  使用 mxnet 编写的 crf 层
│   │   
│   │───lstm_crf.py             LSTM+CRF 算法
│   │   
│   │───predict_flask.py        使用 Flask 运行预测服务
│   │   
│   │───predict.py              使用训练好的模型作为脚本预测
│   │   
│   │───train_helper.py         模型训练时的一些辅助函数
│   │   
│   │───train_model.py          训练指定的模型并保存参数
│   │   
│   │───utils.py                数据处理的一些通用函数
│   │   
│
```
