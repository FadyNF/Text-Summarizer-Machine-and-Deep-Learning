import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import spacy
import torch.optim as optim

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer("all-MiniLM-L6-v2")


def spacy_sent_tokenize(text: str) -> list:
    cleaned_text = text.replace("\n", " ").strip()
    doc = nlp(cleaned_text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def label_sentences_by_rouge(article_sentences, reference_summary, threshold=0.5):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    labels = []
    for sent in article_sentences:
        score = scorer.score(reference_summary, sent)["rouge1"].fmeasure
        labels.append(1 if score >= threshold else 0)
    return labels


def prepare_data(df, max_sents=30):
    X, y = [], []
    for _, row in df.iterrows():
        article = row["Article"]
        summary = row["Summary"]
        sentences = spacy_sent_tokenize(article)[:max_sents]
        labels = label_sentences_by_rouge(sentences, summary)
        embeddings = encoder.encode(sentences)
        X.extend(embeddings)
        y.extend(labels)
    return np.array(X), np.array(y)


class SentenceScorer(nn.Module):
    def __init__(self, input_dim=384):
        super(SentenceScorer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.linear(x)


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())
        acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1} - Val Accuracy: {acc:.4f}")
        report = classification_report(val_targets, val_preds, digits=4)
        print(f"Classification Report:\n{report}")


def get_data_loader(X, y, batch_size=32, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loader(model, data_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    return acc


def summarize_and_evaluate(article, summary, model, top_k=3):
    sentences = spacy_sent_tokenize(article)
    embeddings = encoder.encode(sentences)
    with torch.no_grad():
        logits = model(torch.tensor(embeddings, dtype=torch.float32))
        scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    top_ids = np.argsort(scores)[-top_k:]
    top_ids.sort()
    predicted_summary = " ".join([sentences[i] for i in top_ids])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(summary, predicted_summary)
    return predicted_summary, scores


def run_pipeline(df):
    print("Splitting data (70/15/15)...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Preparing train data...")
    X_train, y_train = prepare_data(train_df)
    print("Preparing validation data...")
    X_val, y_val = prepare_data(val_df)
    print("Preparing test data...")
    X_test, y_test = prepare_data(test_df)

    train_loader = get_data_loader(X_train, y_train)
    val_loader = get_data_loader(X_val, y_val, shuffle=False)
    test_loader = get_data_loader(X_test, y_test, shuffle=False)

    model = SentenceScorer()
    print("Training model...")
    train_model(model, train_loader, val_loader)

    print("\nFinal Accuracies:")
    train_acc = evaluate_loader(model, train_loader)
    val_acc = evaluate_loader(model, val_loader)
    test_acc = evaluate_loader(model, test_loader)
    print(f"Train Accuracy:      {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    print("\nSample Evaluation:")
    for i in range(3):
        article = df.iloc[i]["Article"]
        summary = df.iloc[i]["Summary"]
        pred, rouge = summarize_and_evaluate(article, summary, model)
        print(f"\n📄 Article:\n{article[:300]}...")
        print(f"\n✂️ Predicted Summary:\n{pred}")
        print(f"\n✅ ROUGE:\n{rouge}")

    return model
