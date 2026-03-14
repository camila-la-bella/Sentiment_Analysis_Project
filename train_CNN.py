import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
import csv
import os
import time

# 参数设置
MAX_LEN = 200 #句子最大长度，没有这么长则补齐
PAD_IDX = 0 #PAD补齐默认为0，UNK为随机初始化的词向量
UNK_IDX = 1
BATCH_SIZE = 64 #参数更新时的样本数量
TRAINING_ROUNDS = 10 #训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #使用GPU加速训练，如果没有GPU则使用CPU（本地跑只有CPU）


# 数据集 
class Sentiment_Types_Dataset(Dataset):
    def __init__(self, filepath, word2idx):
        self.texts, self.labels = self.read_data_from_set(filepath)
        self.word2idx = word2idx
        self.data = [self.word_2_wordvec(sentence) for sentence in self.texts]
        print(f"Class distribution: {np.bincount(self.labels)}")

    def read_data_from_set(self, path):
        texts, labels = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                label, *words = line.strip().split()
                texts.append(words)
                labels.append(int(label))
        return texts, labels #将文件读入划为两个部分，一个情感标签0或1，另一个为句子

    def word_2_wordvec(self, tokens):
        ids = [self.word2idx.get(word, UNK_IDX) for word in tokens]
        if len(ids) < MAX_LEN:
            ids += [PAD_IDX] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return ids #将句子中的每个词转换为对应的索引，长度不够则补齐，超过则截断

    def __len__(self):
        return len(self.labels) #返回数据集的长度

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long) #返回数据集中第idx个样本的句子和标签


# CNN_ModelForText 模型
class CNN_ModelForText(nn.Module):
    def __init__(self, embedding_matrix, filter_sizes=(3, 4, 5), num_filters=100, dropout=0.5):
        super().__init__()
        wordvec_size, matrix_dim = embedding_matrix.shape #获取词向量矩阵的大小和维度
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) #使用预训练的词向量矩阵初始化嵌入层，即初始化各个词为“训练起点”
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=matrix_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ]) #尝试使用不同大小的卷积核进行卷积操作，in_channels为输入通道数，out_channels为输出通道数，kernel_size为卷积核大小
        # ModuleList用于存储多个卷积层，filter_sizes为卷积核大小列表
        self.dropout = nn.Dropout(dropout) #防止过拟合
        # dropout层，dropout为丢弃率
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2) #全连接层，num_filters * len(filter_sizes)为输入特征数，2为输出类别数（正面或负面）

    def forward(self, x_dat):
        x_dat = self.embedding(x_dat) # [B, T, D]
        x_dat = x_dat.permute(0, 2, 1) # [B, D, T]，将句子从[B, T, D]转换为[B, D, T]，以便进行卷积操作
        conv_outs = [F.adaptive_max_pool1d(F.relu(conv(x_dat)), 1).squeeze(-1) for conv in self.convs] # [B, num_filters, 1]，先得到卷积后的结果，再对每个卷积层的输出进行pooling操作，得到每个卷积核的最大值
        x_dat = torch.cat(conv_outs, dim=1) ## [B, num_filters * len(filter_sizes)]，将所有卷积核的输出拼接在一起
        x_dat = self.dropout(x_dat) # [B, num_filters * len(filter_sizes)]，对拼接后的输出进行dropout操作，即对每一个值随机置零
        return self.fc(x_dat)


# 验证函数
def mod_evaluation_fun(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch) # [B, 2]，模型输出的logits
            preds = torch.argmax(logits, dim=1) # [B]，获取每个样本的预测类别
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist()) #将预测结果和真实标签都转移到CPU上，并转换为列表形式
    return accuracy_score(all_labels, all_preds) # 计算准确率
    # all_labels为真实标签，all_preds为预测标签


# 网格搜索
if __name__ == "__main__":
    print("Loading word vectors...")
    wv = KeyedVectors.load_word2vec_format("wiki_word2vec_50.bin", binary=True) #加载词向量bin文件
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embedding_dim = wv.vector_size
    embedding_matrix = [np.zeros(embedding_dim), np.random.normal(size=(embedding_dim,))]
    for word in wv.key_to_index:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(wv[word])
    embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)
    print(f"Vocab size: {len(word2idx)}, embedding dim: {embedding_dim}")

    train_dataset = Sentiment_Types_Dataset("data/train.txt", word2idx) #加载训练集
    val_dataset = Sentiment_Types_Dataset("data/validation.txt", word2idx) #加载验证集
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    filter_sizes_list = [(2, 3, 4), (3, 4, 5), (2, 3, 4, 5)] #尝试不同大小的卷积核组合
    dropout_list = [0.3, 0.5, 0.7] #尝试不同的dropout率
    num_filters_list = [64, 100, 150] #尝试不同数量的卷积核
    lr_list = [1e-3, 7e-4, 5e-4] #尝试不同的学习率
    # 超参数组合

    total_configs = len(filter_sizes_list) * len(dropout_list) * len(num_filters_list) * len(lr_list)
    config_counter = 0

    os.makedirs("saved_models", exist_ok=True)
    with open("grid_search_results_CNN.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filters", "dropout", "num_filters", "lr", "best_val_acc"]) #保存网格搜索结果
        print(f"Starting grid search in {total_configs} configurations...")

        for fs, dr, nf, lr in itertools.product(filter_sizes_list, dropout_list, num_filters_list, lr_list):
            config_counter += 1
            print(
                f"[{config_counter}/{total_configs}] Training with filters={fs}, dropout={dr}, num_filters={nf}, lr={lr}")

            model = CNN_ModelForText(embedding_matrix, filter_sizes=fs, num_filters=nf, dropout=dr).to(DEVICE) #初始化模型
            loss_fn = nn.CrossEntropyLoss() #交叉熵损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam优化器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1) #学习率调度器
            # ReduceLROnPlateau根据验证集的准确率动态调整学习率

            best_val_acc = 0.0 # 初始化最佳验证集准确率
            for round in range(TRAINING_ROUNDS): # 训练轮数
                model.train()
                total_loss = 0
                pbar = tqdm(train_loader, desc=f"[Round {round + 1:02d}]", ncols=80)
                for x_batch, y_batch in pbar:
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    logits = model(x_batch) # [B, 2]，模型输出的logits
                    loss = loss_fn(logits, y_batch) #计算损失

                    optimizer.zero_grad() # 梯度清零
                    loss.backward() # 反向传播计算梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # 梯度裁剪，防止梯度爆炸
                    optimizer.step()
                    # 用梯度下降的反向学习传播法，更新参数

                    total_loss += loss.item() * x_batch.size(0) # 计算总损失

                val_acc = mod_evaluation_fun(model, val_loader)  #用验证集计算准确率
                avg_loss = total_loss / len(train_loader.dataset)
                scheduler.step(val_acc) # 更新学习率
                #根据验证集的准确率动态调整学习率
                # 如果验证集准确率没有提升，则降低学习率

                if val_acc > best_val_acc: # 如果验证集准确率提升，则保存模型
                    best_val_acc = val_acc
                    model_path = f"saved_models/model_fs-{str(fs)}_dr-{dr}_nf-{nf}_lr-{lr}.pt"
                    torch.save(model.state_dict(), model_path)

                pbar.close()
                print(f"[Round {round + 1:02d}] Summary: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

            writer.writerow([str(fs), dr, nf, lr, best_val_acc])
            print(
                f"Finished training: filters={fs}, dropout={dr}, num_filters={nf}, lr={lr} | Best Val Acc = {best_val_acc:.4f}")
