import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import sent_tokenize
from rouge import Rouge
import nltk

nltk.download("punkt")


class RandomForestSummarizer:
    def __init__(self, dataset_name, df):
        self.dataset_name = dataset_name
        self.df = df.rename(columns={"Article": "article", "Summary": "summary"})
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.clf = RandomForestClassifier(
            max_depth=30,
            min_samples_leaf=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        )
        self._prepare_data()

    def _prepare_data(self):
        self.sentences, self.labels = [], []
        for _, row in self.df.iterrows():
            sents = sent_tokenize(row["article"])
            summary_sents = set(sent_tokenize(row["summary"]))
            self.sentences.extend(sents)
            self.labels.extend([1 if s in summary_sents else 0 for s in sents])

    def _extract_summary(self, article):
        sents = sent_tokenize(article)
        X = self.vectorizer.transform(sents)
        preds = self.clf.predict(X)
        extracted = [s for s, p in zip(sents, preds) if p == 1]
        return " ".join(extracted) if extracted else sents[0]

    def run(self):
        X = self.vectorizer.fit_transform(self.sentences)
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.labels, test_size=0.3, random_state=42
        )

        print(f"\n=== Running on {self.dataset_name} Dataset ===")
        self.clf.fit(X_train, y_train)

        # Training reports
        print(f"Train Accuracy: {self.clf.score(X_train, y_train):.4f}")
        print(f"Test Accuracy: {self.clf.score(X_test, y_test):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, self.clf.predict(X_test)))

        # ROUGE evaluation
        sample_df = self.df.sample(n=200, random_state=42)
        rouge = Rouge()
        scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

        for _, row in sample_df.iterrows():
            pred = self._extract_summary(row["article"])
            true = row["summary"]
            try:
                score = rouge.get_scores(pred, true)[0]
                for k in scores:
                    scores[k].append(score[k]["f"])
            except:
                continue

        print("\nROUGE Scores:")
        for k, v in scores.items():
            print(f"{k}: {sum(v)/len(v):.4f}")

    def show_samples(self, n=5):
        sample_df = self.df.sample(n=n, random_state=1)
        for i, row in sample_df.iterrows():
            print(f"\n--- Article {i+1} ---")
            print("Predicted:", self._extract_summary(row["article"]))
            print("Reference:", row["summary"])
            print("-" * 80)
