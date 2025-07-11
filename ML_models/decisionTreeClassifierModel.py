import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
from rouge import Rouge


class DecisionTreeClassifierModel:
    def __init__(self, dataset_name, df):
        self.dataset_name = dataset_name
        self.df = df.copy()

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True
        )

        self.clf = DecisionTreeClassifier(
            max_depth=30,
            min_samples_leaf=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
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

        # 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        self.clf.fit(X_train, y_train)

        y_pred_test = self.clf.predict(X_test)
        y_pred_val = self.clf.predict(X_val)

        print(f"\nTrain Accuracy: {self.clf.score(X_train, y_train):.4f}")
        print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred_test))

    def show_predictions(self, n=5):
        rouge = Rouge()
        article_ids = sorted(self.df['article_id'].unique())[:n]

        all_rouge_scores = []

        print("\n=== Article-wise Summary Evaluation ===")
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
                    all_rouge_scores.append(scores)
                except Exception as e:
                    print(f"Error computing ROUGE: {e}")

        # Compute average ROUGE scores
        if all_rouge_scores:
            avg_scores = {}
            for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                avg_scores[metric] = {
                    k: round(
                        sum(score[metric][k] for score in all_rouge_scores) / len(all_rouge_scores), 4
                    )
                    for k in ['p', 'r', 'f']
                }

            print("\n=== Average ROUGE Scores ===")
            for metric, values in avg_scores.items():
                print(f"{metric.upper()}: Precision: {values['p']}, Recall: {values['r']}, F1: {values['f']}")
        else:
            print("\nCould not compute average ROUGE scores (no valid articles).")