# 处理生成的results下的csv; 1、去除双O生成check.scv； 2、根据check生成check_word.csv; (慎重，会覆盖check和check_word.csv)

import os
import glob
import csv
import shutil


def read_csv(filename):
    data = []
    prev_row = None
    with open(filename, 'r', newline='', encoding="utf_8_sig") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if prev_row is not None and prev_row[1] == "O" and "I-" in row[1]:  # 前为O的替换
                row = [row[0], row[1].replace("I-", "B-"), row[2]]
            if row[1] != "O" or row[2] != "O":
                data.append(row)
            prev_row = row
    # print(data)
    return data


def get_labels(data):
    i = 0
    labels = []
    while i < len(data):
        row = data[i]
        if row[2] == "B-Color" or row[2] == "B-Nature":
            label, i = get_label(data, row[2], i)
            # print(label)
            labels.append(label)
        else:
            label, i = get_Olabel(data, row[2], i)
            # print(label)
            labels.append(label)
    return labels


def get_Plabels(data):
    i = 0
    labels = []
    while i < len(data):
        row = data[i]
        if row[1] == "B-Color" or row[1] == "B-Nature":
            label, i = get_Plabel(data, row[1], i)
            # print(label)
            labels.append(label)
        else:
            i += 1
    return labels


def get_label(data, type, index):
    current_word = [data[index][0]]  # 用于构建当前的单词
    mytype = "Nature"
    pre = "TP"
    if type == "B-Color":
        mytype = "Color"
    while index < len(data):
        if data[index][1] != data[index][2] and data[index][1] != "O" and (pre != "FN" or mytype not in data[index][1]):
            pre = "FP"  # 错误识别
        elif data[index][1] != data[index][2] and data[index][1] == "O" and pre != "FP":
            pre = "FN"  # 未能正确识别
        index += 1
        if index < len(data) and data[index][2] == "I-" + mytype:
            current_word.append(data[index][0])
        elif index < len(data) and data[index][2] != "I-" + mytype and pre == "TP" and data[index][1] == "I-" + mytype:
            current_word.append(data[index][0])
            if data[index][2] == "O":
                pre = "FP"  # 错误识别
            else:
                pre = "FN"  # 未能正确识别
        else:
            break
    return [''.join(current_word), pre, mytype], index


def get_Plabel(data, type, index):
    current_word = [data[index][0]]  # 用于构建当前的单词
    mytype = "Nature"
    if type == "B-Color":
        mytype = "Color"
    while index < len(data):
        index += 1
        if index < len(data) and data[index][1] == "I-" + mytype:
            current_word.append(data[index][0])
        else:
            break
    return [''.join(current_word), "Wait", mytype], index


def get_Olabel(data, type, index):
    current_word = [data[index][0]]  # 用于构建当前的单词
    while index < len(data):
        index += 1
        if index < len(data) and data[index][2] == "O":
            current_word.append(data[index][0])
        else:
            break
    return [''.join(current_word), "FP", "O"], index  # 错误识别


def write_csv(filename, ret):
    with open(filename, 'w', newline='', encoding="utf_8_sig") as csvfile:
        csvwriter = csv.writer(csvfile)
        for rows in ret:
            csvwriter.writerow(rows)


# data/predict/final目录下是根据results生成的；check.csv是去除O之后的, check_word是合成的单词（需要核查的）
def make_csv(type):
    result_dir = 'data/prepare/results'
    final_dir = 'data/prepare/final'
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    # 使用glob获取目录中所有的CSV文件
    csv_files = glob.glob(os.path.join(result_dir, type + '*.csv'))

    # 对文件进行排序
    csv_files.sort()  # reverse=True
    for i, file in enumerate(csv_files):
        # print("第" + str(i) + "轮*" * 100)
        data = read_csv(file)
        write_csv(final_dir + "/check_" + type + str(i) + ".csv", data)
        if type == "test":
            labels = get_labels(data)
        else:
            labels = get_Plabels(data)
        write_csv(final_dir + "/check_" + type + "_word" + str(i) + ".csv", labels)
        # cal(labels)
        # print("*" * 100)


def make_csvs():
    if os.path.exists('data/prepare/final'):
        shutil.rmtree('data/prepare/final')
    os.makedirs('data/prepare/final')
    make_csv("test")
    make_csv("predict")


# 打印的是最终的结果
if __name__ == '__main__':
    make_csvs()
