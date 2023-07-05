import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset


def padding(poetries, maxlen, pad):
    batch_seq = [poetry + pad * (maxlen - len(poetry)) for poetry in poetries]
    return batch_seq


# 输入向后滑一字符为target，即预测下一个字
def split_input_target(seq):
    inputs = seq[:-1]
    targets = seq[1:]
    return inputs, targets


# 创建词汇表
def get_poetry(arg):
    poetrys = []
    if arg.Augmented_dataset:
        path = arg.Augmented_data
    else:
        path = arg.data
    with open(path, "r", encoding='UTF-8') as f:
        for line in f:
            try:
                # line = line.decode('UTF-8')
                line = line.strip(u'\n')
                if arg.Augmented_dataset:
                    content = line.strip(u' ')
                else:
                    title, content = line.strip(u' ').split(u':')
                content = content.replace(u' ', u'')
                if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:
                    continue
                if arg.strict_dataset:
                    if len(content) < 12 or len(content) > 79:
                        continue
                else:
                    if len(content) < 5 or len(content) > 79:
                        continue
                content = u'[' + content + u']'
                poetrys.append(content)
            except Exception as e:
                pass

            # 按诗的字数排序
    poetrys = sorted(poetrys, key=lambda line: len(line))

    with open("data/org_poetry.txt", "w", encoding="utf-8") as f:
        for poetry in poetrys:
            poetry = str(poetry).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'
            f.write(poetry)

    return poetrys


# 切分文档
def split_text(poetrys):
    with open("data/split_poetry.txt", "w", encoding="utf-8") as f:
        for poetry in poetrys:
            poetry = str(poetry).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n '
            split_data = " ".join(poetry)
            f.write(split_data)
    return open("data/split_poetry.txt", "r", encoding='UTF-8').read()


# 训练词向量
def train_vec(split_file="data/split_poetry.txt", org_file="data/org_poetry.txt"):
    param_file = "data/word_vec.pkl"
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")
    if os.path.exists(split_file):
        all_data_split = open(split_file, "r", encoding="utf-8").read().split("\n")
    else:
        all_data_split = split_text().split("\n")

    if os.path.exists(param_file):
        return org_data, pickle.load(open(param_file, "rb"))

    models = Word2Vec(all_data_split, vector_size=256, workers=7, min_count=1)
    pickle.dump([models.syn1neg, models.wv.key_to_index, models.wv.index_to_key], open(param_file, "wb"))
    return org_data, (models.syn1neg, models.wv.key_to_index, models.wv.index_to_key)


class Poetry_Dataset(Dataset):
    def __init__(self, w1, word_2_index, all_data, Word2Vec, model):
        self.model=model
        self.Word2Vec = Word2Vec
        self.w1 = w1
        self.word_2_index = word_2_index
        word_size, embedding_num = w1.shape
        self.embedding = nn.Embedding(word_size, embedding_num)
        # 最长句子长度
        maxlen = max([len(seq) for seq in all_data])
        pad = ' '
        self.all_data = padding(all_data[:-1], maxlen, pad)

    def __getitem__(self, index):
        a_poetry = self.all_data[index]
        if self.model=="bart":
            a_poetry_padded=a_poetry.replace(" ","[PAD]")
            # xs, ys = split_input_target(a_poetry)
            from transformers import BertTokenizer
            tokenizer=BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
            # xs_padded=xs.replace(" ","[PAD]")
            # ys_padded=ys.replace(" ","[PAD]")
            # xs=tokenizer(xs_padded,return_tensors="pt")["input_ids"][0]
            # ys=tokenizer(ys_padded,return_tensors="pt")["input_ids"][0]
            a_poetry_index=tokenizer(a_poetry_padded,return_tensors="pt")["input_ids"][0]
        if self.model=="bert":
            a_poetry_padded=a_poetry.replace(" ","[PAD]")
            # xs, ys = split_input_target(a_poetry)
            from transformers import BertTokenizer
            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
            a_poetry_index=tokenizer(a_poetry_padded,return_tensors="pt")["input_ids"][0]
        else:
            a_poetry_index = [self.word_2_index[i] for i in a_poetry]
        xs, ys = split_input_target(a_poetry_index)
        if self.Word2Vec:
            xs_embedding = self.w1[xs]
        else:
            xs_embedding = np.array(xs)

        return xs_embedding, np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)
