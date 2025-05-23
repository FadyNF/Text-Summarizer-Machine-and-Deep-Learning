# ML_models/cnn.py

import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from rouge import Rouge


class CNNExtractiveSummarizer:
    def __init__(self, max_vocab=5000, max_len=50, embedding_dim=100, random_state=42):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.tokenizer = Tokenizer(num_words=self.max_vocab, oov_token="<OOV>")
        self.rouge = Rouge()
        self.model = None

    def _build_model(self):
        model = Sequential(
            [
                Embedding(
                    input_dim=self.max_vocab,
                    output_dim=self.embedding_dim,
                    input_length=self.max_len,
                ),
                Conv1D(filters=128, kernel_size=5, activation="relu"),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                # Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def _prepare_input(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences, maxlen=self.max_len, padding="post", truncating="post"
        )

    def tune(self, X_train_raw, y_train, X_val_raw=None, y_val=None, epochs=5):
        self.tokenizer.fit_on_texts(X_train_raw)
        X_train = self._prepare_input(X_train_raw)
        self.model = self._build_model()

        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        if X_val_raw is not None and y_val is not None:
            X_val = self._prepare_input(X_val_raw)
            self.model.fit(
                X_train,
                np.array(y_train),
                validation_data=(X_val, np.array(y_val)),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
            )
        else:
            self.model.fit(
                X_train,
                np.array(y_train),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
            )

    def train(self, X_train_raw, y_train):
        self.tokenizer.fit_on_texts(X_train_raw)
        X_train = self._prepare_input(X_train_raw)
        self.model = self._build_model()
        self.model.fit(X_train, np.array(y_train), epochs=5, batch_size=32)

    def predict(self, X_raw):
        X = self._prepare_input(X_raw)
        preds = self.model.predict(X)
        return (preds > 0.5).astype(int).flatten()

    def predict_proba(self, X_raw):
        X = self._prepare_input(X_raw)
        return self.model.predict(X)

    def evaluate(self, X_test_raw, y_test):
        y_pred = self.predict(X_test_raw)
        print(classification_report(y_test, y_pred))

    def summarize(
        self, article_sentences, preprocessed_sentences, top_n=3, threshold=0.5
    ):
        if not preprocessed_sentences:
            return "No summary generated"
        probas = self.predict_proba(preprocessed_sentences).flatten()
        confident_indices = [i for i, p in enumerate(probas) if p >= threshold]

        if not confident_indices:
            top_indices = np.argsort(probas)[-top_n:]
        else:
            top_indices = sorted(
                confident_indices, key=lambda i: probas[i], reverse=True
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
