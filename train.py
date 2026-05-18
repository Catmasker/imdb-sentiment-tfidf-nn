"""
IMDB 情感分析 —— GloVe 词嵌入 + 神经网络分类器
============================================
工作流程：
  1. 加载 IMDB 影评数据 (imdb_balanced_10k.csv)
  2. 加载 tiny_glove.json 词向量（每个词对应一个 50 维向量）
  3. 将每条评论的所有词向量取平均 → 得到一个 50 维的"评论向量"
  4. 用这个 50 维向量训练一个全连接神经网络做二分类（好评/差评）
  5. 保存模型、配置、指标到 model/ 目录
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ──────────────────────────────────────────────
# 1. 设置路径
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 2. 加载数据
# ──────────────────────────────────────────────
print("=" * 60)
print("加载数据...")
df = pd.read_csv(os.path.join(DATA_DIR, "imdb_balanced_10k.csv"))
print(f"数据集大小: {len(df)} 条")
print(f"列名: {df.columns.tolist()}")
print(f"好评数: {(df['label'] == 1).sum()}, 差评数: {(df['label'] == 0).sum()}")

# ──────────────────────────────────────────────
# 3. 文本预处理函数
# ──────────────────────────────────────────────
def clean_text(text):
    """清洗文本：去 HTML 标签、去特殊字符、转小写"""
    text = re.sub(r"<br\s*/?>", " ", text)       # 去掉 <br> 标签
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # 只保留字母和空格
    text = text.lower()                           # 统一小写
    return text

def tokenize(text):
    """分词：按空格切分"""
    return text.split()

print("\n预处理文本...")
df["clean_text"] = df["text"].apply(clean_text)
df["tokens"] = df["clean_text"].apply(tokenize)

# 打印一条预处理前后的对比
print("\n预处理示例：")
print(f"  原文:  {df['text'].iloc[0][:150]}...")
print(f"  清洗后: {df['clean_text'].iloc[0][:150]}...")
print(f"  分词数: {len(df['tokens'].iloc[0])} 个词")

# ──────────────────────────────────────────────
# 4. 加载 GloVe 词嵌入（tiny_glove.json）
# ──────────────────────────────────────────────
print("\n加载 GloVe 词向量...")
with open(os.path.join(DATA_DIR, "tiny_glove.json"), "r") as f:
    glove = json.load(f)

EMBEDDING_DIM = len(next(iter(glove.values())))  # 向量维度（应该是 50）
print(f"词向量总数: {len(glove)} 个词")
print(f"向量维度: {EMBEDDING_DIM} 维")

print(f"示例: 'movie' → 前 5 维 = {glove.get('movie', ['N/A'])[:5]}")

# ──────────────────────────────────────────────
# 5. 将每条评论转换为 100 维向量（均值池化 + 最大值池化）
# ──────────────────────────────────────────────
def review_to_vector(tokens, glove, dim=50):
    """
    核心思想：把评论里每个词查 GloVe 向量，然后拼接两个统计量：
      - mean: 所有词向量的平均值 → 捕捉"整体语义"
      - max:  所有词向量的逐维最大值 → 捕捉"最显著特征"
    两者拼接得到 50+50=100 维向量。
    """
    vectors = [glove[word] for word in tokens if word in glove]
    if len(vectors) == 0:
        return np.zeros(dim * 2)
    vecs = np.array(vectors)
    mean_pool = np.mean(vecs, axis=0)
    max_pool = np.max(vecs, axis=0)
    return np.concatenate([mean_pool, max_pool])

print("\n将评论转换为词向量（均值+最大值池化）...")
X = np.array([review_to_vector(tokens, glove, EMBEDDING_DIM) for tokens in df["tokens"]])
y = df["label"].values.astype(np.float32)

INPUT_DIM = X.shape[1]
print(f"特征矩阵形状: {X.shape}  (每条评论 → {INPUT_DIM} 维向量 = 50维均值 + 50维最大值)")

# ──────────────────────────────────────────────
# 6. 划分训练集 / 测试集
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ──────────────────────────────────────────────
# 7. 构建神经网络模型
# ──────────────────────────────────────────────
class SentimentMLP(nn.Module):
    """
    多层感知机 (MLP, Multilayer Perceptron)

    输入层 (100)        ← 50 维 GloVe 均值 + 50 维 GloVe 最大值
        ↓
    全连接 (100 → 256) + BatchNorm + ReLU + Dropout(0.3)
        ↓
    全连接 (256 → 128) + BatchNorm + ReLU + Dropout(0.3)
        ↓
    全连接 (128 → 64)  + BatchNorm + ReLU + Dropout(0.2)
        ↓
    输出层 (64 → 1) + Sigmoid → 好评概率 [0, 1]

    BatchNorm: 对每层的输出做归一化，稳定训练、加速收敛
    Dropout:   随机丢弃部分神经元，防止过拟合
    """
    def __init__(self, input_dim=100, hidden1=256, hidden2=128, hidden3=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = SentimentMLP(input_dim=INPUT_DIM)
print(f"\n模型结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")

# ──────────────────────────────────────────────
# 8. 设置训练参数
# ──────────────────────────────────────────────
criterion = nn.BCELoss()          # 二分类交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=False
)                                 # 当准确率不再提升时，学习率减半
epochs = 100                      # 训练轮数
batch_size = 64                   # 每批 64 条

print(f"\n训练配置:")
print(f"  损失函数: BCELoss (二元交叉熵)")
print(f"  优化器: Adam (学习率=0.001, 自适应衰减)")
print(f"  训练轮数: {epochs}")
print(f"  批次大小: {batch_size}")

# ──────────────────────────────────────────────
# 9. 训练循环
# ──────────────────────────────────────────────
n = len(X_train_t)
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # Mini-batch 训练：每次取 batch_size 条数据
    for i in range(0, n, batch_size):
        X_batch = X_train_t[i:i+batch_size]
        y_batch = y_train_t[i:i+batch_size]

        # 前向传播: 输入 → 模型 → 预测
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播: 计算梯度 → 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 每个 epoch 结束后在测试集上评估
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_bin = (test_preds.numpy() > 0.5).astype(int)
        test_acc = accuracy_score(y_test, test_preds_bin)

    # 学习率调度：如果准确率不再提升，学习率减半
    scheduler.step(test_acc)

    train_losses.append(epoch_loss / (n // batch_size))
    test_accuracies.append(test_acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
              f"Loss: {train_losses[-1]:.4f}  "
              f"Test Acc: {test_acc:.4f}  "
              f"LR: {lr:.6f}")

# ──────────────────────────────────────────────
# 10. 最终评估
# ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_t).numpy().flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("最终测试集评估结果:")
print(f"  Accuracy (准确率):  {accuracy:.4f}")
print(f"  Precision (精确率): {precision:.4f}")
print(f"  Recall (召回率):    {recall:.4f}")
print(f"  F1 Score:           {f1:.4f}")

# ──────────────────────────────────────────────
# 11. 保存模型和配置
# ──────────────────────────────────────────────
print("\n保存模型到 model/ 目录...")

# 保存 PyTorch 模型权重
model_path = os.path.join(MODEL_DIR, "model.pt")
torch.save(model.state_dict(), model_path)
print(f"  ✓ 模型权重 → {model_path}")

# 保存预处理配置（prediction 时需要知道输入维度、用哪个词表）
config = {
    "input_dim": INPUT_DIM,
    "embedding_dim": EMBEDDING_DIM,
    "pooling": "mean+max (concatenated)",
    "model_architecture": "SentimentMLP(100→256→128→64→1) + BatchNorm",
    "dataset": "imdb_balanced_10k.csv",
    "embedding_source": "tiny_glove.json",
    "test_accuracy": float(accuracy),
    "threshold": 0.5,
}
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"  ✓ 配置     → {config_path}")

# 保存词向量（predict.py 需要用它把新文本转成向量）
glove_path = os.path.join(MODEL_DIR, "glove_embedding.pkl")
with open(glove_path, "wb") as f:
    pickle.dump(glove, f)
print(f"  ✓ 词向量   → {glove_path}")

# 保存评估指标
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "test_size": len(X_test),
    "train_size": len(X_train),
}
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ 指标     → {metrics_path}")

print(f"\n✅ 训练完成！所有文件已保存到 {MODEL_DIR}")
print(f"\n测试集准确率: {accuracy:.2%}")
