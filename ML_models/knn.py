import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from rouge import Rouge


class KNNExtractiveSummarizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rouge = Rouge()
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = ImbPipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("ros", RandomOverSampler(random_state=self.random_state)),
                ("clf", KNeighborsClassifier()),
            ]
        )

    def tune(self, X_train, y_train, n_iter=10, scoring="f1"):
        param_distributions = {
            "tfidf__max_features": [3000, 5000, 7000, None],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "clf__n_neighbors": [1, 3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["cosine", "euclidean"],
        }

        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=3,
            verbose=0,
            random_state=self.random_state,
            n_jobs=1,
        )

        search.fit(X_train, y_train)
        self.pipeline = search.best_estimator_
        print("Best params:", search.best_params_)
        print("Best score:", search.best_score_)

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test):
        try:
            return self.pipeline.predict_proba(X_test)
        except AttributeError:
            print(
                "KNN doesn't support predict_proba with cosine distance. Switching to euclidean or using confidence hack."
            )
            return np.zeros((len(X_test), 2))  # fallback

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))

    def summarize(
        self, article_sentences, preprocessed_sentences, top_n=3, threshold=0.1
    ):
        if not preprocessed_sentences:
            return "No summary generated"
        try:
            probas = self.predict_proba(preprocessed_sentences)
            prob_class1 = probas[:, 1]
        except:
            print("Falling back to label prediction for sentence selection")
            preds = self.predict(preprocessed_sentences)
            prob_class1 = np.where(preds == 1, 1.0, 0.0)

        confident_indices = [i for i, p in enumerate(prob_class1) if p >= threshold]

        if not confident_indices:
            top_indices = np.argsort(prob_class1)[-top_n:]
        else:
            top_indices = sorted(
                confident_indices, key=lambda i: prob_class1[i], reverse=True
            )[:top_n]

        top_indices = sorted(top_indices)
        selected_sents = [
            article_sentences[i] for i in top_indices if i < len(article_sentences)
        ]

        return " ".join(selected_sents) if selected_sents else "No summary generated"

    def compute_rouge(self, generated_summary, reference_summary):
        try:
            return self.rouge.get_scores(generated_summary, reference_summary)
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return None
