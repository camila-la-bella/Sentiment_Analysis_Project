import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import gensim

# 参数设置
MAX_LEN = 200
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 情感数据类
class SentimentDataset(Dataset):
    def __init__(self, filepath, word2idx):
        self.texts, self.labels = self.read_data(filepath)
        self.word2idx = word2idx
        self.data = [self.encode(sentence) for sentence in self.texts]

    def read_data(self, path):
        texts, labels = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                label, *words = line.strip().split()
                texts.append(words)
                labels.append(int(label))
        return texts, labels

    def encode(self, tokens):
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
class MLPClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)       # [B, T, D]
        x = x.mean(dim=1)           # 平均池化 → [B, D]
        x = torch.relu(self.fc1(x)) # → [B, H]
        x = self.dropout(x)
        return self.fc2(x)          # → [B, 2]

# 计算 Precision, Recall, F1 的函数
def compute_metrics(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 / (1/precision + 1/recall) if precision + recall else 0

    return precision, recall, f1

# accuracy评估函数
def evaluate_metrics(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1 = compute_metrics(all_labels, all_preds)
    return acc, f1, precision, recall

if __name__ == "__main__":
    print("Loading word vectors...")
    wv = gensim.models.KeyedVectors.load_word2vec_format("wiki_word2vec_50.bin", binary=True)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embedding_dim = wv.vector_size
    embedding_matrix = [np.zeros(embedding_dim), np.random.normal(size=(embedding_dim,))]
    for word in wv.key_to_index:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(wv[word])
    embedding_matrix = torch.tensor(np.array(embedding_matrix), dtype=torch.float)

    # 加载测试集
    test_dataset = SentimentDataset("data/test.txt", word2idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 加载模型，注意修改参数以匹配模型
    model_path = "saved_models_res/model_MLP_hd-64_dr-0.3_lr-0.001.pt"
    model = MLPClassifier(embedding_matrix, hidden_dim=64, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 评估并打印结果
    acc, f1, prec, rec = evaluate_metrics(model, test_loader)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
