from data_process import *
from prepare_data import *
from data_utils import *
from train import *
from process_csv import *
from precision import *
# 一：运行步骤：
# 首先运行start.py(处理predict、train文本)；
# 然后新增或修改data/train/20230810/predict1.txt，predict2.txt...;
# 再运行startPredict.py(单独对新增或修改predict文本进行处理)


# dataMain(): 对应prepare_data.py（本质运行一次就行了）
# utilMain(): 对于data_utils.py（根据csv生成test,train,predict的pkl）
# train(): 对于data_train.py（根据四个pkl进行训练，预测）
# make_csvs() 处理生成的results下的csv; 1、去除双O生成check.scv； 2、根据check生成check_word.csv; (慎重，会覆盖check和check_word.csv)
# cal_csvs()  读取check_word.csv，计算三个公式


# 目录介绍：
# data目录：包含train和prepare目录。train目录是训练的原始数据（predict.txt是需要预测的原始文本）；prepare目录是代码生成的
# data/predict/results目录下是train生成的，共十轮；test.csv是测试集的结果；predict是预测集
# data/prepare目录: 包含train、test、predict目录，分别存放生成的训练、测试、预测csv，用于生成pkl。
# data/prepare/results目录: 保存的train.py预测的完整结果，包含测试test.csv和预测predict.csv，结构为文本、预测、实际标签（如霞、I-Color、O）
# data/prepare/final目录: 保存的处理后的csv，分为测试和预测两类。去除双O生成的check.csv；组合单词生成实体的check_word.csv(可用于计算指标);
# data/prepare/*.pkl: 通过utilMain()，根据csv生成test,train,predict的pkl
# data/prepare/*.txt: 把pkl打印成txt


if __name__ == '__main__':
    dataMain()  # 根据txt,生成train、test和predict目录下的csv以及dict.pkl（重新运行会删除prepare目录）
    utilMain()  # 根据csv生成test,train,predict的pkl
    train()  # 根据四个pkl进行训练，预测得到results下的csv  (有四个的pkl后，可单独运行)
    make_csvs()  # 处理生成的results下的csv; 1、去除双O生成check.scv； 2、根据check生成check_word.csv; (慎重，会覆盖check和check_word.csv)
    cal_csvs()  # 读取check_word.csv，计算三个公式
