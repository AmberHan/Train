from data_process import *
from prepare_data import *
from data_utils import *
from train import *
from process_csv import *
from precision import *

# 如果需要自己的预测；运行将需要的预测的csv放到data/predict目录下
# data/predict/results目录下是train生成的，共十轮；test.csv是测试集的结果；predict是预测集

# dataMain: 对应prepare_data.py（本质运行一次就行了）
# utilMain: 对于data_utils.py（根据csv生成test,train,predict的pkl）
# train: 对于data_train.py（根据四个pkl进行训练，预测）
#
# 1、首先运行prepare_data.py（生成train和test目录下的csv以及dict.pkl（重新运行会删除prepare目录））；
# 2、然后将你想预测的csv拷贝到predict目录下；
# 3、先后执行data_utils.py、data_train.py即可。

if __name__ == '__main__':
    dataMain()  # 生成train和test目录下的csv以及dict.pkl（重新运行会删除prepare目录）
    utilMain()  # 根据csv生成test,train,predict的pkl
    train()  # 根据四个pkl进行训练，预测得到results下的csv  (有四个的pkl后，可单独运行)
    make_csv()  # 处理生成的results下的csv; 1、去除双O生成check.scv； 2、根据check生成check_word.csv; (慎重，会覆盖check和check_word.csv)
    cal_csv()  # 读取check_word.csv，计算三个公式
