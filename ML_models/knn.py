from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def train_knn_on_dataset(df, dataset_name=""):
    print(f"\nTraining on {dataset_name} dataset...")

    # Prepare data
    X_texts = df["preprocessed_sentence"].values
    y = df["label"].values

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_texts)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Evaluation
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    return knn, vectorizer
