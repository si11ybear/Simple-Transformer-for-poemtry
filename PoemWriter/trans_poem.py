import math
import numpy as np
from collections import Counter
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import random
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import os
import argparse
from ..simpleTransformer.classes import * 

def preprocess_data(data_path, max_len=64, min_len=5, disallowed_words=None, min_word_frequency=1):
    if disallowed_words is None:
        disallowed_words = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']', '？', '；']

    poetry = []

    # 按行读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        fields = line.split(":")
        if len(fields) != 2:
            continue
        content = fields[1].strip()  # 去掉换行符和前后空格

        # 过滤数据
        if len(content) > max_len - 2 or len(content) < min_len:
            continue
        if any(word in content for word in disallowed_words):
            continue
        poetry.append(content.replace('\n', ''))

    counter = Counter()
    for line in poetry:
        counter.update(line)
    tokens = [token for token, count in counter.items() if count >= min_word_frequency]

    return poetry, tokens

# 参数配置
EPOCH_NUM = 50
BATCH_SIZE = 64  # 内存不够的话，就把BATCH_SIZE调小点
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(DATA_PATH):
    poetry, tokens = preprocess_data(DATA_PATH)

    # 实例化 Tokenizer
    tokenizer = Tokenizer(tokens)
    v = len(tokenizer)
    max_len = 128

    # 实例化MyDataset
    my_dataset = MyDataset(poetry, tokenizer)
    train_dataloader = DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、优化器和损失函数
    h = 128
    a = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 4 * h
    dropout = 0.1

    model = Transformer(v, h, a, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    #print(model)

    for epoch in range(1, EPOCH_NUM + 1):
        model.train()
        total_loss = 0
        data_progress = tqdm.tqdm(train_dataloader, desc="Train...")
        for step, data in enumerate(data_progress, 1):
            data = data.to(DEVICE)
            # 随机选一个位置，拆分src和tgt
            e = random.randint(1, 20)
            src = data[:, :e].to(DEVICE)
            # tgt不要最后一个token，tgt_y不要第一个的token
            tgt, tgt_y = data[:, e:-1].to(DEVICE), data[:, e + 1:].to(DEVICE)

            # 进行Transformer的计算，再将结果送给最后的线性层进行预测
            out = model(src, tgt)
            out = model.predict(out)
            loss = criterion(out.view(-1, out.size(-1)), tgt_y.contiguous().view(-1))

            with torch.autograd.set_detect_anomaly(False):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            # 更新训练进度
            data_progress.set_description(f"Train... [epoch {epoch}/{EPOCH_NUM}, loss {(total_loss / step):.5f}]")

    # 从 DATA_PATH 获取文件名
    base_name = os.path.splitext(os.path.basename(DATA_PATH))[0]

    if torch.cuda.is_available():
        # 保存两个版本模型
        torch.save(model.state_dict(), f'model/{base_name}_model.pth')
        model.cpu()
        torch.save(model.state_dict(), f'model/{base_name}_model_cpu.pth')
    else:
        torch.save(model.state_dict(), f'model/{base_name}_model_cpu.pth')

    # 保存 tokenizer
    with open(f'model/{base_name}_tokenizer_tokens.txt', 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/tangshi.txt', help='Path to the input text file')
    args = parser.parse_args()
    main(args.data_path)

    #python trans_poem.py --data_path data/tangshi.txt
