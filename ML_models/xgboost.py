import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

class XGBoostClassifierModel:
    def __init__(self, dataset_name, df):
        self.dataset_name = dataset_name
        self.df = df
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def run(self):
        print(f"\n=== Running classification on {self.dataset_name} Dataset ===")

        # Extract data
        self.X = self.df["text"]
        self.y = self.df["label"]

        # Vectorize text
        self.X_tfidf = self.vectorizer.fit_transform(self.X)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_tfidf, self.y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(self.X_train, self.y_train)

        # Predict
        self.y_pred = self.model.predict(self.X_test)

        # Metrics
        train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        test_acc = accuracy_score(self.y_test, self.y_pred)

        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred))

    def show_classification_predictions(self, n=5):
        print(f"\n\n=== Classification Predictions for {self.dataset_name} ===")

        if not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            print("⚠️ Model hasn't been trained yet. Please run model.run() first.")
            return

        sample_indices_loc = np.random.choice(len(self.y_test), size=min(n, len(self.y_test)), replace=False)
        original_df_indices = self.y_test.index[sample_indices_loc]

        for i, df_idx in enumerate(original_df_indices):
            try:
                test_set_idx = self.y_test.index.get_loc(df_idx)
                pred_label = self.y_pred[test_set_idx]
                true_label = self.y_test.loc[df_idx]

                print(f"\n--- Article ID: {df_idx} ---")
                text_preview = str(self.df.loc[df_idx, 'text'])[:300]
                print(f"Text Preview: {text_preview}...")
                print(f"Predicted Label: {pred_label}")
                print(f"True Label: {true_label}")
                print(f"Correct: {'✓' if pred_label == true_label else '✗'}")

                if hasattr(self.model, 'predict_proba'):
                    try:
                        test_sample = self.X_test[test_set_idx]
                        probabilities = self.model.predict_proba(test_sample)
                        max_prob = np.max(probabilities)
                        print(f"Confidence: {max_prob:.3f}")
                    except:
                        pass

            except KeyError:
                print(f"⚠️ Article ID {df_idx} not found. Skipping.")
                continue

    def show_predictions(self, n=5):
        self.show_classification_predictions(n)