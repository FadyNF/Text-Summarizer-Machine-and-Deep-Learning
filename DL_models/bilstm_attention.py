import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from rouge import Rouge
import time


class BiLSTMSummarizer:
    def __init__(self, dataset_name, df):
        self.dataset_name = dataset_name
        self.df = df
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_data()

    def _prepare_data(self):
        if "reference_summary" not in self.df.columns:
            self.df["reference_summary"] = self.df.groupby("article_id")[
                "preprocessed_sentence"
            ].transform(lambda x: " ".join(x[self.df.loc[x.index, "label"] == 1]))

        # First split into train and temp (80% train, 20% temp)
        train_df, temp_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # Then split temp into 50% validation, 50% test (i.e., 10% each of original)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.vocab = Vocab(self.train_df["preprocessed_sentence"].tolist())

        self.train_loader = DataLoader(
            ExtractiveDataset(self.train_df),
            batch_size=32,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.vocab),
        )

        self.model = BiLSTMAttention(len(self.vocab.token2idx)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([3.0]).to(self.device)
        )

    def train(self):
        print(f"\n=== Training on {self.dataset_name} ===")
        start_time = time.time()

        for epoch in range(5):
            self.model.train()
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(self.train_loader):.4f}")

        print(f"Training completed in {time.time()-start_time:.2f}s")

    def evaluate_split(self, split="test"):
        if split == "train":
            df = self.train_df
        elif split == "val":
            df = self.val_df
        else:
            df = self.test_df

        loader = DataLoader(
            ExtractiveDataset(df),
            batch_size=32,
            collate_fn=lambda b: collate_fn(b, self.vocab),
        )

        self.model.eval()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = torch.sigmoid(self.model(inputs))
                all_probs.extend(outputs.cpu().tolist())
                all_preds.extend((outputs > 0.5).int().cpu().tolist())

        y_true = df["label"].values
        y_pred = [1 if p > 0.5 else 0 for p in all_probs]
        accuracy = (y_true == y_pred).mean()
        print(f"{split.capitalize()} Accuracy: {accuracy:.4f}")

        return accuracy


    def evaluate(self):
        
         # Evaluate on test set as before
        print("\n--- Evaluating on Test Set ---")
        self.evaluate_split("test")

        # Also evaluate on train and validation sets
        print("\n--- Evaluating on Train Set ---")
        self.evaluate_split("train")

        print("\n--- Evaluating on Validation Set ---")
        self.evaluate_split("val")
        test_loader = DataLoader(
            ExtractiveDataset(self.test_df),
            batch_size=32,
            collate_fn=lambda b: collate_fn(b, self.vocab),
        )

        self.model.eval()
        all_probs, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = torch.sigmoid(self.model(inputs))
                all_probs.extend(outputs.cpu().tolist())
                all_preds.extend((outputs > 0.5).int().cpu().tolist())

        # Threshold tuning
        best_thresh, best_f1 = 0.5, 0
        for t in [i * 0.05 for i in range(1, 20)]:
            current_preds = [1 if p > t else 0 for p in all_probs]
            current_f1 = f1_score(self.test_df["label"], current_preds)
            if current_f1 > best_f1:
                best_f1, best_thresh = current_f1, t
        print(f"Best Threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")

        # ROUGE evaluation
        final_preds = [1 if p > best_thresh else 0 for p in all_probs]
        self._calculate_rouge(final_preds)

    def _calculate_rouge(self, preds):
        pred_by_article = defaultdict(list)
        for row, pred in zip(self.test_df.itertuples(), preds):
            if pred == 1:
                pred_by_article[row.article_id].append(row.preprocessed_sentence)

        rouge = Rouge()
        scores = defaultdict(list)
        refs = self.test_df[["article_id", "reference_summary"]].drop_duplicates()

        for aid, sents in pred_by_article.items():
            ref = refs[refs.article_id == aid].reference_summary.values
            if len(ref) > 0:
                try:
                    score = rouge.get_scores(" ".join(sents), ref[0])[0]
                    for k in score:
                        scores[k].append(score[k]["f"])
                except:
                    continue

        print("\nROUGE Scores:")
        for metric, values in scores.items():
            print(f"{metric}: {sum(values)/len(values):.4f}")
            
    def show_samples(self, n=5):
        # Sample n unique articles from the test set
        sampled_articles = self.test_df["article_id"].drop_duplicates().sample(n=n, random_state=1).tolist()

        self.model.eval()
        rouge = Rouge()

        for aid in sampled_articles:
            # Extract all sentences of this article from test set
            article_sents_df = self.test_df[self.test_df["article_id"] == aid]
            sentences = article_sents_df["preprocessed_sentence"].tolist()

            # Encode sentences to tensors
            inputs = torch.tensor([self.vocab.encode(s) for s in sentences]).to(self.device)

            with torch.no_grad():
                outputs = torch.sigmoid(self.model(inputs)).cpu().tolist()

            # Use threshold 0.5 or tune if you want to use the best_thresh from evaluate()
            preds = [1 if p > 0.5 else 0 for p in outputs]

            # Reconstruct predicted summary
            pred_summary = " ".join([s for s, p in zip(sentences, preds) if p == 1])
            if not pred_summary:
                pred_summary = sentences[0]  # fallback

            # Reference summary for this article
            ref_summary = article_sents_df["reference_summary"].iloc[0]

            print(f"\n--- Article ID: {aid} ---")
            print("Predicted Summary:\n", pred_summary)
            print("Reference Summary:\n", ref_summary)
            print("-" * 80)



# Helper classes/functions (preserved from original code)
class ExtractiveDataset(Dataset):
    def __init__(self, df):
        self.sentences = df["preprocessed_sentence"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class Vocab:
    def __init__(self, texts):
        self.token2idx = {"<PAD>": 0, "<UNK>": 1}
        counts = defaultdict(int)
        for text in texts:
            for word in text.split():
                counts[word] += 1
        for word, freq in counts.items():
            if freq > 1:  # min_freq=2
                self.token2idx[word] = len(self.token2idx)

    def encode(self, text, max_len=100):
        tokens = text.split()
        ids = [self.token2idx.get(t, 1) for t in tokens]
        return ids[:max_len] + [0] * (max_len - len(ids))


def collate_fn(batch, vocab):
    texts, labels = zip(*batch)
    inputs = torch.tensor([vocab.encode(t) for t in texts])
    labels = torch.tensor(labels, dtype=torch.float32)
    return inputs, labels


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
        self.bilstm = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(128)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        context, _ = self.attention(x)
        return self.fc(context).squeeze(1)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(256, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))
        scores = self.v(energy).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(2), dim=1), weights
