from data_process import *
from prepare_data import *
from data_utils import *
from train import *
from process_csv import *
from precision import *

if __name__ == '__main__':
    dataMain(True)  # 根据data/train/20230810/predict.txt,生成predict目录下的csv
    utilMain(True)  # 根据csv生成predict的pkl
    train()  # 根据四个pkl进行训练
    make_csvs()  # 处理生成的results下的csv; 1、去除双O生成check.scv； 2、根据check生成check_word.csv; (慎重，会覆盖check和check_word.csv)
    cal_csvs()  # 读取check_word.csv，计算三个公式