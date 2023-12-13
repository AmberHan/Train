# 读取check_word.csv，计算三个公式

import os
import glob
import csv


def read_csv(filename):
    data = []
    with open(filename, 'r', newline='', encoding="utf_8_sig") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    # print(data)
    return data


def cal(labels):
    # 实际的实体类别
    # 错误识别：如果实际是O,但模型预测为实体;一个为color，一个为nature,则是FP
    # 未能正确：如果实际是实体,但模型预测为O，则是FN
    # actual_labels = ["B-Nature", "I-Nature", "B-Color", "I-Color", "O"]

    # 初始化FN和FP的计数器
    FN = 0  # 假负例
    FP = 0  # 假正例
    TP = 0  # 真正例
    TN = 0  # 真负例

    # 计算FN和FP
    for label in labels:
        if label[1] == "FN":
            FN += 1
        elif label[1] == "FP":
            FP += 1
        elif label[1] == "TP":
            TP += 1

    # 计算精确度
    precision = TP / (TP + FP)

    # 计算召回率
    recall = TP / (TP + FN)

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)

    # ACC，TN暂时为0
    acc = (TP + TN) / (TP + FP + FN + TN)

    # 打印FN、FP、精确度、召回率和F1分数
    print("拼音+词边界：")  # 词性+词边界+偏旁部首+拼音：
    print("真正例（TP）:", TP)
    print("假负例（FN）:", FN)
    print("假正例（FP）:", FP)
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)
    print("ACC:", acc)


def write_csv(filename, ret):
    with open(filename, 'w', newline='', encoding="utf_8_sig") as csvfile:
        csvwriter = csv.writer(csvfile)
        for rows in ret:
            csvwriter.writerow(rows)


# data/predict/final目录下是根据results生成的；check.csv是去除O之后的, check_word是合成的单词（需要核查的）
def cal_csv(type):
    final_dir = 'data/prepare/final'
    # 使用glob获取目录中所有的CSV文件
    csv_files = glob.glob(os.path.join(final_dir, type + '*.csv'))

    # 对文件进行排序
    csv_files.sort()  # reverse=True
    for i, file in enumerate(csv_files):
        print("第" + str(i) + "轮*" + "*" * 100)
        data = read_csv(file)
        cal(data)
        print("*" * 100)


def cal_csvs():
    print("test集合" + "*" * 200)
    cal_csv("check_test_word")
    # print("预测集合" + "*" * 200)
    # cal_csv("check_predict_word")


# 打印的是最终的结果
if __name__ == '__main__':
    cal_csvs()
