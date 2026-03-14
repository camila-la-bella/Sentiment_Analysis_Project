import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score
import numpy as np
import itertools
import csv
import os
from tqdm import tqdm

#  基本参数同CNN模型
MAX_LEN = 200
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 64
TRAINING_ROUNDS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
class Sentiment_Types_Dataset(Dataset):
    def __init__(self, filepath, word2idx): #加载数据集，同CNN模型
        self.texts, self.labels = self.read_data_from_set(filepath)
        self.word2idx = word2idx
        self.data = [self.word_to_vec(sentence) for sentence in self.texts]

    def read_data_from_set(self, path):
        texts, labels = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                label, *words = line.strip().split()
                texts.append(words)
                labels.append(int(label))
        return texts, labels

    def word_to_vec(self, tokens):
        ids = [self.word2idx.get(word, UNK_IDX) for word in tokens]
        if len(ids) < MAX_LEN:
            ids += [PAD_IDX] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 双向 LSTM 模型 
class LSTM_BiForText(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.5):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) #加载预训练词向量
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True) #双向LSTM
        self.dropout = nn.Dropout(dropout)#防止过拟合
        self.fc = nn.Linear(hidden_dim * 2, 2) #二分类，双向 LSTM 的输出是 [B, H*2]，所以这里要乘 2

    def forward(self, x):
        x = self.embedding(x)  # [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: [2, B, H]，最后一个时间步的隐藏状态
        h_forward = h_n[-2]
        h_backward = h_n[-1] # 取双向 LSTM 的最后一个时间步的隐藏状态
        h = torch.cat((h_forward, h_backward), dim=1)  # [B, H*2]，拼接正向和反向的隐藏状态
        out = self.dropout(h)
        return self.fc(out)

# 验证评估函数，同CNN模型
def mod_evaluation_fun(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())
    return accuracy_score(all_labels, all_preds)

# 主程序
if __name__ == "__main__":
    print("Loading word vectors...")
    wv = KeyedVectors.load_word2vec_format("wiki_word2vec_50.bin", binary=True) #加载词向量bin文件 
    word2idx = {"<PAD>": 0, "<UNK>": 1} #PAD补齐默认为0，UNK为随机初始化的词向量
    embedding_dim = wv.vector_size
    embedding_matrix = [np.zeros(embedding_dim), np.random.normal(size=(embedding_dim,))]
    for word in wv.key_to_index:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(wv[word])
    embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

    # 加载数据
    train_dataset = Sentiment_Types_Dataset("data/train.txt", word2idx)
    val_dataset = Sentiment_Types_Dataset("data/validation.txt", word2idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 超参数组合
    hidden_dim_list = [64, 128, 256] #隐藏层维度
    dropout_list = [0.3, 0.5, 0.7] #dropout比率
    lr_list = [1e-3, 7e-4, 5e-4] #初始学习率

    total_configs = len(hidden_dim_list) * len(dropout_list) * len(lr_list)
    config_counter = 0

    os.makedirs("saved_models", exist_ok=True)
    with open("grid_search_results_LSTM_bidirectional.csv", mode="w", newline="", encoding="utf-8") as f: #保存结果
        writer = csv.writer(f)
        writer.writerow(["hidden_dim", "dropout", "lr", "best_val_acc"]) 

        for hd, dr, lr in itertools.product(hidden_dim_list, dropout_list, lr_list):
            config_counter += 1
            print(f"[{config_counter}/{total_configs}] hidden_dim={hd}, dropout={dr}, lr={lr}")

            model = LSTM_BiForText(embedding_matrix, hidden_dim=hd, dropout=dr).to(DEVICE)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

            best_val_acc = 0.0
            for epoch in range(TRAINING_ROUNDS):
                model.train()
                total_loss = 0
                pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1:02d}]", ncols=80)
                for x_batch, y_batch in pbar:
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    logits = model(x_batch)
                    loss = loss_fn(logits, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += loss.item() * x_batch.size(0)
                    #仍然使用梯度下降法反向传播，同CNN模型

                val_acc = mod_evaluation_fun(model, val_loader)
                avg_loss = total_loss / len(train_loader.dataset)
                print(f"[Epoch {epoch + 1}] Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
                scheduler.step(val_acc) # 学习率调整

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = f"saved_models/model_LSTM_bi_hd-{hd}_dr-{dr}_lr-{lr}.pt"
                    torch.save(model.state_dict(), model_path)

            writer.writerow([hd, dr, lr, best_val_acc])
            print(f">>> Best val acc = {best_val_acc:.4f} <<<\n")
