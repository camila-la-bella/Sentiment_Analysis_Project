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
class Sentiment_LSTM_Dataset(Dataset):
    def __init__(self, filepath, word2idx):
        self.texts, self.labels = self.read_data(filepath)
        self.word2idx = word2idx
        self.data = [self.encode(sentence) for sentence in self.texts]

    def read_data(self, path): #读取数据
        texts, labels = [], []
        with open(path, encoding='utf-8') as f:
            for line in f:
                label, *words = line.strip().split()
                texts.append(words)
                labels.append(int(label))
        return texts, labels

    def encode(self, tokens): #词语转换为词向量
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

# LSTM 模型类
class LSTM_ModelForText(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 因为双向LSTM所以输出维度乘2

    def forward(self, x):
        x = self.embedding(x)                         # [B, T, D]
        lstm_out, (h_n, _) = self.lstm(x)             # h_n: [2, B, H]
        h_forward = h_n[-2]                           # 正向最后时刻
        h_backward = h_n[-1]                          # 反向最后时刻
        h = torch.cat((h_forward, h_backward), dim=1) # 拼接正反向隐藏状态 [B, 2H]
        return self.fc(self.dropout(h))               # [B, 2]

# 计算 Precision, Recall, F1 的函数
def calculate_f1_precision_recall(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

# 模型评估
def evaluating_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds) # 计算模型准确率
    precision, recall, f1 = calculate_f1_precision_recall(all_labels, all_preds)
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

    model_path = "saved_models_res/model_LSTM_bi_hd-64_dr-0.3_lr-0.001.pt" #使用 grid search 得到的最优model进行验证
    model = LSTM_ModelForText(embedding_matrix).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_dataset = Sentiment_LSTM_Dataset("data/test.txt", word2idx) #使用 test data 进行模型验证
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    acc, f1, prec, rec = evaluating_model(model, test_loader)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
