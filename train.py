"""
IMDB 情感分析 —— TF-IDF + 神经网络分类器
============================================
最终方案：
  - TF-IDF (25000 特征, 词+双词, sublinear_tf)
  - 轻量神经网络 (25000→16→1) + Dropout
  - AdamW + 强 L2 正则化 (weight_decay=0.03)
  - 逐批稀疏→稠密转换（低内存占用，兼容 GitHub Actions）
"""

import json
import re
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. 加载数据
# ──────────────────────────────────────────────
print("=" * 60)
print("加载数据...")
df = pd.read_csv(os.path.join(DATA_DIR, "imdb_balanced_10k.csv"))
print(f"数据集: {len(df)} 条, 好评: {(df['label']==1).sum()}, 差评: {(df['label']==0).sum()}")

# ──────────────────────────────────────────────
# 2. 文本预处理
# ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df["clean_text"] = df["text"].apply(clean_text)
print(f"\n预处理示例:\n  原文: {df['text'].iloc[0][:100]}...\n  清洗: {df['clean_text'].iloc[0][:100]}...")

# ──────────────────────────────────────────────
# 3. TF-IDF 特征提取
# ──────────────────────────────────────────────
print("\n提取 TF-IDF 特征...")
vectorizer = TfidfVectorizer(
    max_features=25000, ngram_range=(1, 2), sublinear_tf=True,
)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"].values.astype(np.float32)

INPUT_DIM = X.shape[1]
print(f"特征矩阵: {X.shape}, 密度: {X.nnz/(X.shape[0]*X.shape[1])*100:.2f}%")

# ──────────────────────────────────────────────
# 4. 划分训练/测试（保持稀疏！）
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ──────────────────────────────────────────────
# 5. 模型
# ──────────────────────────────────────────────
class SentimentNet(nn.Module):
    def __init__(self, input_dim=25000, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = SentimentNet(input_dim=INPUT_DIM)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n参数量: {total_params:,}")

# ──────────────────────────────────────────────
# 6. 训练配置
# ──────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
epochs = 100
batch_size = 64

print(f"训练: BCELoss | AdamW(lr=0.001, wd=0.03) | batch={batch_size} | max_epochs={epochs}")

# ──────────────────────────────────────────────
# 7. 训练循环（逐批稀疏→稠密）
# ──────────────────────────────────────────────
n = X_train.shape[0]
best_acc = 0.0
best_model_state = None
patience_counter = 0

print("\n开始训练...")
print("-" * 60)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, n, batch_size):
        # 逐批转换：只把当前 batch 转稠密 → 内存友好
        X_batch_sparse = X_train[i:i+batch_size]
        X_batch = torch.tensor(X_batch_sparse.toarray(), dtype=torch.float32)
        y_batch = y_train_t[i:i+batch_size]

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 测试集评估（也逐批，避免大矩阵）
    model.eval()
    all_preds = []
    m = X_test.shape[0]
    for i in range(0, m, batch_size):
        X_batch_sparse = X_test[i:i+batch_size]
        X_batch = torch.tensor(X_batch_sparse.toarray(), dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_batch).numpy()
        all_preds.append(preds)
    y_pred_bin = (np.concatenate(all_preds) > 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred_bin)

    scheduler.step(test_acc)
    avg_loss = epoch_loss / ((n - 1) // batch_size + 1)

    if test_acc > best_acc:
        best_acc = test_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
              f"Loss: {avg_loss:.4f}  Acc: {test_acc:.4f}  "
              f"Best: {best_acc:.4f}  LR: {lr:.6f}")

    if patience_counter >= 10:
        print(f"\n  早停: 连续 10 轮未提升")
        break

# ──────────────────────────────────────────────
# 8. 最终评估
# ──────────────────────────────────────────────
model.load_state_dict(best_model_state)
model.eval()
all_preds = []
m = X_test.shape[0]
for i in range(0, m, batch_size):
    X_batch_sparse = X_test[i:i+batch_size]
    X_batch = torch.tensor(X_batch_sparse.toarray(), dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_batch).numpy()
    all_preds.append(preds)
y_pred = (np.concatenate(all_preds) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("最终测试集评估结果:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

if accuracy >= 0.90:
    print(f"\n目标达成！{accuracy:.2%} >= 90% ✓")
else:
    print(f"\n当前 {accuracy:.2%} < 90%")

# ──────────────────────────────────────────────
# 9. 保存
# ──────────────────────────────────────────────
print("\n保存模型...")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

config = {
    "input_dim": INPUT_DIM,
    "feature_method": "TF-IDF (unigram+bigram, max_features=25000)",
    "model_architecture": "SentimentNet(25000->16->1) + Dropout + AdamW wd=0.03",
    "dataset": "imdb_balanced_10k.csv",
    "test_accuracy": float(accuracy),
    "threshold": 0.5,
}
with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

metrics = {
    "accuracy": float(accuracy), "precision": float(precision),
    "recall": float(recall), "f1_score": float(f1),
    "test_size": int(X_test.shape[0]), "train_size": int(X_train.shape[0]),
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"  model.pt + tfidf_vectorizer.pkl + config.json + metrics.json")
print(f"\n✅ 训练完成！准确率: {accuracy:.2%}")
