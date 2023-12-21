from prepare_data import *
from data_utils import *
from train import *

if __name__ == '__main__':
    dataMain(True)  # 根据data/train/20230810/predict.txt,生成predict目录下的csv
    utilMain(True)  # 根据csv生成predict的pkl
    train(True)  # 根据四个pkl进行训练
