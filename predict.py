"""
预测脚本 —— 用训练好的模型对新评论做情感分析
用法:
  python predict.py "This movie was fantastic!"
  python predict.py "What a terrible film."
"""

import json
import re
import pickle
import sys
import torch
import torch.nn as nn


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


def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()

    model.eval()
    with torch.no_grad():
        x = torch.tensor(vec, dtype=torch.float32)
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

    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    model = SentimentNet(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model.pt"), map_location="cpu"
    ))

    if len(sys.argv) < 2:
        test_texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "What a waste of time. The acting was terrible and the plot made no sense.",
            "It was okay, not great but not terrible either.",
        ]
        print("没有提供文本，使用示例:")
        for text in test_texts:
            result = predict(text, model, vectorizer)
            print(f"\n  文本: {result['text']}")
            print(f"  情感: {result['sentiment'].upper()}")
            print(f"  好评概率: {result['positive_prob']}")
    else:
        text = " ".join(sys.argv[1:])
        result = predict(text, model, vectorizer)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment'].upper()}")
        print(f"好评概率: {result['positive_prob']}")


if __name__ == "__main__":
    main()
