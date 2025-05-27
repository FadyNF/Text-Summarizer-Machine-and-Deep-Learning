import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
from rouge import Rouge
import numpy as np


class RandomForestClassifierModel:
    def __init__(self, dataset_name, df):
        self.dataset_name = dataset_name
        self.df = df.copy()

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True
        )

        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    def _add_features(self, df):
        df = df.copy()
        df['sentence_length'] = df['preprocessed_sentence'].apply(lambda x: len(x.split()))
        df['sentence_position'] = df.groupby('article_id').cumcount() + 1
        df['sentence_position'] = df['sentence_position'] / df.groupby('article_id')['preprocessed_sentence'].transform('count')
        return df

    def _balance_data(self, df):
        majority = df[df['label'] == 0]
        minority = df[df['label'] == 1]
        if len(minority) == 0:
            return df  # avoid crash if no summaries
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )
        return pd.concat([majority, minority_upsampled])

    def run(self):
        print(f"\n=== Running classification on {self.dataset_name} Dataset ===")

        df = self._add_features(self.df)
        df = self._balance_data(df)

        X_text = self.vectorizer.fit_transform(df['preprocessed_sentence'].astype(str))
        X_additional = df[['sentence_length', 'sentence_position']].values
        X_all = hstack([X_text, csr_matrix(X_additional)]).tocsr()
        y = df['label'].values

        # Split 80% train, 10% val, 10% test
        X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.10, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp)
        # 0.1111 of 90% = ~10%

        self.clf.fit(X_train, y_train)

        print(f"\nTrain Accuracy: {self.clf.score(X_train, y_train):.4f}")
        print(f"Validation Accuracy: {self.clf.score(X_val, y_val):.4f}")
        print(f"Test Accuracy: {self.clf.score(X_test, y_test):.4f}")
        print("\nTest Classification Report:")
        y_pred = self.clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def show_predictions(self, n=5):
        rouge = Rouge()
        article_ids = sorted(self.df['article_id'].unique())[:n]

        print("\n=== Article-wise Summary Evaluation ===")
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []

        for article_id in article_ids:
            article_df = self.df[self.df['article_id'] == article_id].copy()
            article_df = self._add_features(article_df)

            X_text = self.vectorizer.transform(article_df['preprocessed_sentence'].astype(str))
            X_additional = article_df[['sentence_length', 'sentence_position']].values
            X_all = hstack([X_text, csr_matrix(X_additional)]).tocsr()

            preds = self.clf.predict(X_all)

            ref_sents = article_df[article_df['label'] == 1]['article_sentences'].tolist()
            gen_sents = article_df[preds == 1]['article_sentences'].tolist()

            ref_summary = " ".join(ref_sents).strip()
            gen_summary = " ".join(gen_sents).strip()

            print(f"\nArticle ID: {article_id}")
            print(f"Reference Summary: {ref_summary if ref_summary else '...'}")
            print(f"Generated Summary: {gen_summary if gen_summary else '...'}")

            if not ref_summary or not gen_summary:
                print("Error computing ROUGE: Reference or generated summary is empty.")
            else:
                try:
                    scores = rouge.get_scores(gen_summary, ref_summary)[0]
                    rounded_scores = {k: {m: round(v, 4) for m, v in scores[k].items()} for k in scores}
                    print(f"ROUGE Scores: {rounded_scores}")

                    rouge_1_scores.append(scores['rouge-1']['f'])
                    rouge_2_scores.append(scores['rouge-2']['f'])
                    rouge_l_scores.append(scores['rouge-l']['f'])

                except Exception as e:
                    print(f"Error computing ROUGE: {e}")

        # Show average ROUGE scores
        if rouge_1_scores:
            print("\n=== Average ROUGE Scores ===")
            print(f"ROUGE-1-F: {np.mean(rouge_1_scores):.4f}")
            print(f"ROUGE-2-F: {np.mean(rouge_2_scores):.4f}")
            print(f"ROUGE-L-F: {np.mean(rouge_l_scores):.4f}")
        else:
            print("\nNo valid summaries for ROUGE evaluation.")