"""
预测脚本 —— 用训练好的模型对新评论做情感分析
=============================================
用法:
  python predict.py "This movie was fantastic!"
  python predict.py "This movie was terrible."

流程:
  1. 加载训练好的模型权重和配置
  2. 加载 GloVe 词向量和 TF-IDF 向量器
  3. 对输入文本做同样的预处理 → TF-IDF 加权 GloVe 平均 + 最大值池化
  4. 输入模型 → 输出好评概率
"""

import json
import re
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 模型定义（必须与 train.py 完全一致）
# ──────────────────────────────────────────────
class SentimentMLP(nn.Module):
    """与训练时完全相同的网络结构"""
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


# ──────────────────────────────────────────────
# 文本预处理（与 train.py 一致）
# ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

def tokenize(text):
    return text.split()


def review_to_vector(tokens, glove, dim=50):
    """与训练时相同的均值+最大值池化"""
    vectors = [glove[word] for word in tokens if word in glove]
    if len(vectors) == 0:
        return np.zeros(dim * 2)
    vecs = np.array(vectors)
    mean_pool = np.mean(vecs, axis=0)
    max_pool = np.max(vecs, axis=0)
    return np.concatenate([mean_pool, max_pool])


def predict(text, model, glove, config):
    """对一段文本做情感预测。"""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    vec = review_to_vector(tokens, glove, dim=config["embedding_dim"])

    model.eval()
    with torch.no_grad():
        x = torch.tensor(vec, dtype=torch.float32).view(1, -1)
        prob = model(x).item()

    return {
        "text": text,
        "positive_prob": round(prob, 4),
        "sentiment": "positive" if prob > 0.5 else "negative",
    }


def main():
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    # 加载配置
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # 加载词向量
    glove_path = os.path.join(MODEL_DIR, "glove_embedding.pkl")
    with open(glove_path, "rb") as f:
        glove = pickle.load(f)

    # 加载模型
    model = SentimentMLP(input_dim=config["input_dim"])
    model_path = os.path.join(MODEL_DIR, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # 处理输入
    if len(sys.argv) < 2:
        test_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "What a waste of time. The acting was terrible and the plot made no sense.",
            "It was okay, not great but not terrible either.",
        ]
        print("没有提供文本，使用示例:")
        for text in test_texts:
            result = predict(text, model, glove, config)
            print(f"\n  文本: {result['text']}")
            print(f"  情感: {result['sentiment'].upper()}")
            print(f"  好评概率: {result['positive_prob']}")
    else:
        text = " ".join(sys.argv[1:])
        result = predict(text, model, glove, config)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment'].upper()}")
        print(f"好评概率: {result['positive_prob']}")


if __name__ == "__main__":
    main()
