import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import spacy
import torch.optim as optim

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Load pretrained encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def spacy_sent_tokenize(text: str) -> list:
    """
    Splits text into sentences using spaCy.
    Normalizes line breaks first.
    """
    cleaned_text = text.replace("\n", " ").strip()
    doc = nlp(cleaned_text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# 1. Sentence-level ROUGE labeling (weak supervision)
def label_sentences_by_rouge(article_sentences, reference_summary, threshold=0.5):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    labels = []
    for sent in article_sentences:
        score = scorer.score(reference_summary, sent)['rouge1'].fmeasure
        labels.append(1 if score >= threshold else 0)
    return labels

# 2. Prepare dataset
def prepare_data(df, max_sents=30):
    X, y = [], []
    for _, row in df.iterrows():
        article = row['Article']
        summary = row['Summary']
        sentences = spacy_sent_tokenize(article)[:max_sents]  # truncate long articles
        labels = label_sentences_by_rouge(sentences, summary)

        embeddings = encoder.encode(sentences)
        X.extend(embeddings)
        y.extend(labels)
    return np.array(X), np.array(y)

# 3. Model
class SentenceScorer(nn.Module):
    def __init__(self, input_dim=384):  # MiniLM-L6-v2 has 384-dim
        super(SentenceScorer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # binary classification
        )

    def forward(self, x):
        return self.linear(x)

# 4. Training loop
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

        # Compute accuracy
        acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1} - Val Accuracy: {acc:.4f}")

        # Generate classification report
        report = classification_report(val_targets, val_preds, digits=4)
        print(f"Classification Report:\n{report}")
        
        
# 5. Build loaders
def get_data_loaders(X, y, batch_size=32):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(val_set, batch_size=batch_size)

# 6. Inference and ROUGE evaluation
def summarize_and_evaluate(article, summary, model, top_k=3):
    sentences = spacy_sent_tokenize(article)
    embeddings = encoder.encode(sentences)
    with torch.no_grad():
        logits = model(torch.tensor(embeddings, dtype=torch.float32))
        scores = torch.softmax(logits, dim=1)[:, 1].numpy()

    top_ids = np.argsort(scores)[-top_k:]
    top_ids.sort()
    predicted_summary = " ".join([sentences[i] for i in top_ids])

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary, predicted_summary)

    return predicted_summary, scores

# 7. Main pipeline
def run_pipeline(df):
    print("Preparing data...")
    X, y = prepare_data(df)
    train_loader, val_loader = get_data_loaders(X, y)

    model = SentenceScorer()
    print("Training model...")
    train_model(model, train_loader, val_loader)

    print("\nSample Evaluation:")
    for i in range(3):
        article = df.iloc[i]['Article']
        summary = df.iloc[i]['Summary']
        pred, rouge = summarize_and_evaluate(article, summary, model)
        print(f"\n📄 Article:\n{article[:300]}...")
        print(f"\n✂️ Predicted Summary:\n{pred}")
        print(f"\n✅ ROUGE:\n{rouge}")

    return model
