import pickle as pkl
train_iter = 'data/prepare/dict.pkl'
# 保存

# 一次性读取
f = open(train_iter, 'rb')
content = pkl.load(f, encoding='latin1')

s = str(content)
with open(f'data/prepare/test.txt', 'w', encoding='UTF-8') as f:
    f.write(s)