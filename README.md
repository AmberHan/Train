# 基于神经网络的昆曲特征提取

## 目录介绍：
- data目录：包含train和prepare目录。train目录是训练的原始数据（predict.txt是需要预测的原始文本）；prepare目录是代码生成的
- data/predict/results目录下是train生成的，共十轮；test.csv是测试集的结果；predict是预测集
- data/prepare目录: 包含train、test、predict目录，分别存放生成的训练、测试、预测csv，用于生成pkl。
- data/prepare/results目录: 保存的train.py预测的完整结果，包含测试test.csv和预测predict.csv，结构为文本、预测、实际标签（如B-Nature、I-Color、O）
- data/prepare/final目录: 保存的处理后的csv，分为测试和预测两类。去除双O生成的check.csv；组合单词生成实体的check_word.csv(可用于计算指标);
- data/prepare/*.pkl: 通过utilMain()，根据csv生成test,train,predict的pkl
- data/prepare/*.txt: 把pkl打印成txt

## data_process.py
    数据预处理，处理txt和ann文件得到实体csv文件

## data_utils.py
    数据增强预处理，处理pkl文件，通过拼接句子增加长句子学习能力

## train.py
    双层神经网络：
    - 1.输入层（包括字符、边界、标志、偏旁部首和拼音等特征） 
    - 2.嵌入层（每个特征的ID都通过嵌入层转换为固定长度的向量表示）
    - 3.双层双向LSTM（每层包括一个前向LSTM和一个后向LSTM） 
    - 4.输出映射层（全连接层将输入映射到实体）
    train(True)  # only preict 加载model文件夹里面的模型
    train(False) # train and predict 保存模型到model里面
    
## startPredict.py
- 单纯处理预测集

## start.py
- 处理train & test & predict

## precision.py
- 验证集的评估函数
