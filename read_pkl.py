import pickle as pkl


def pkl_to_txt(s_pkl):
    # 一次性读取
    train_iter = 'data/prepare/{}.pkl'.format(s_pkl)
    train_txt = 'data/prepare/{}.txt'.format(s_pkl)
    f = open(train_iter, 'rb')
    content = pkl.load(f, encoding='latin1')
    s = str(content)
    with open(train_txt, 'w', encoding='UTF-8') as f:
        f.write(s)


def pkls_to_txt(onlypredict=False):
    if not onlypredict:
        pkl_to_txt("test")
        pkl_to_txt("train")
        pkl_to_txt("dict")
    pkl_to_txt("predict")


# 打印的是最终的结果
if __name__ == '__main__':
    pkls_to_txt()
