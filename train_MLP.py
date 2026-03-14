import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import gensim
import csv
import itertools
import os
from tqdm import tqdm

# 参数设置
MAX_LEN = 200
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #在有GPU的时候可用GPU加速，但是这个实验中只用了CPU

# 情感数据类
class SentimentDataset(Dataset):
    def __init__(self, filepath, word2idx):
        self.texts, self.labels = self.read_data_from_dataset(filepath)
        self.word2idx = word2idx
        self.data = [self.word_2_vectors(sentence) for sentence in self.texts]

    def read_data_from_dataset(self, path):
        texts, labels = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                label, *words = line.strip().split()
                texts.append(words)
                labels.append(int(label))
        return texts, labels

    def word_2_vectors(self, tokens):
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

# MLP模型类
class MLP_TextProcessing(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.5):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)# [B, T, D]
        x = x.mean(dim=1)# 平均池化 → [B, D]， 这里对每一个句子进行一个词向量的平均
        x = torch.relu(self.fc1(x))# 全连接 + 激活 → [B, H]
        x = self.dropout(x)
        return self.fc2(x)# 输出 logits → [B, 2]

# 评估函数
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

if __name__ == "__main__":
    print("Loading word vectors...")
    word_vec = gensim.models.KeyedVectors.load_word2vec_format("wiki_word2vec_50.bin", binary=True)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embedding_dim = word_vec.vector_size
    embedding_matrix = [np.zeros(embedding_dim), np.random.normal(size=(embedding_dim,))]
    for word in word_vec.key_to_index:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(word_vec[word])
    embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

    train_dataset = SentimentDataset("data/train.txt", word2idx)
    val_dataset = SentimentDataset("data/validation.txt", word2idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    hidden_dim_list = [64, 128, 256] #隐藏层的数量
    dropout_list = [0.3, 0.5, 0.7] #dropout 率
    lr_list = [1e-3, 7e-4, 5e-4] #学习率

    total_configs = len(hidden_dim_list) * len(dropout_list) * len(lr_list)
    config_counter = 0

    os.makedirs("saved_models", exist_ok=True)
    with open("grid_search_results_MLP.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_dim", "dropout", "lr", "best_val_acc"])

        for hd, drl, lr in itertools.product(hidden_dim_list, dropout_list, lr_list):
            config_counter += 1
            print(f"[{config_counter}/{total_configs}] hidden_dim={hd}, dropout={drl}, lr={lr}")

            model = MLP_TextProcessing(embedding_matrix, hidden_dim=hd, dropout=drl).to(DEVICE)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

            best_val_acc = 0.0
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1:02d}]", ncols=80)
                for x_batch, y_batch in pbar:
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    logits = model(x_batch) #计算二分类各个类别的可能性
                    loss = loss_fn(logits, y_batch) #计算损失函数

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += loss.item() * x_batch.size(0)
                    # 用梯度下降的反向学习传播法，更新参数

                val_acc = mod_evaluation_fun(model, val_loader) #计算准确率
                avg_loss = total_loss / len(train_loader.dataset)
                print(f"[Epoch {epoch + 1}] Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc #只保留10次训练中准确率最高的模型
                    model_path = f"saved_models/model_MLP_hd-{hd}_dr-{drl}_lr-{lr}.pt"
                    torch.save(model.state_dict(), model_path)

            writer.writerow([hd, drl, lr, best_val_acc])
            print(f">>> Best val acc = {best_val_acc:.4f} <<<\n")
