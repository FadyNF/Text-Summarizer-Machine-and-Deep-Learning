import os
import nltk
import numpy as np
import math
import pandas as pd
from gensim import corpora, models
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import kagglehub
from kagglehub import KaggleDatasetAdapter
import string
from rouge_score import rouge_scorer

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# Load datasets
def load_datasets(limit_train=10000, limit_val=2000, limit_test=2000):
    print("Loading datasets...")
    train_df = (
        kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "gowrishankarp/newspaper-text-summarization-cnn-dailymail",
            "cnn_dailymail/train.csv",
        )
        .dropna()
        .head(limit_train)
    )
    val_df = (
        kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "gowrishankarp/newspaper-text-summarization-cnn-dailymail",
            "cnn_dailymail/validation.csv",
        )
        .dropna()
        .head(limit_val)
    )
    test_df = (
        kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "gowrishankarp/newspaper-text-summarization-cnn-dailymail",
            "cnn_dailymail/test.csv",
        )
        .dropna()
        .head(limit_test)
    )
    return train_df, val_df, test_df


# Preprocessing function
def preprocess_text(text):
    """
    1. Sentence Tokenization
    2. Word Tokenization
    3. Removal of Stopwords
    """
    # 1. Sentence Tokenization
    sentences = sent_tokenize(text)

    # 2 & 3. Word Tokenization and Stopword Removal
    stop_words = set(stopwords.words("english"))
    processed_sentences = []
    raw_sentences = []

    for sentence in sentences:
        raw_sentences.append(sentence)
        # Remove punctuation and convert to lowercase
        sentence = sentence.lower()
        sentence = "".join(
            [char for char in sentence if char not in string.punctuation]
        )

        # Tokenize words
        words = word_tokenize(sentence)

        # Remove stopwords
        filtered_words = [word for word in words if word not in stop_words]
        processed_sentences.append(filtered_words)

    return raw_sentences, processed_sentences


# LDA for Topic Modeling
def apply_lda(processed_sentences, num_topics=5):
    """
    Apply Latent Dirichlet Allocation for topic modeling
    """
    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(text) for text in processed_sentences]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        alpha="auto",
        eta="auto",
    )
    
    # Save both model and dictionary
    os.makedirs("extended_textrank/saved_model", exist_ok=True)
    lda_model.save("extended_textrank/saved_model/lda_model")
    dictionary.save("extended_textrank/saved_model/lda_dictionary.dict")


    # Get topic distribution for each sentence
    topic_matrix = np.zeros((len(processed_sentences), num_topics))

    for i, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            topic_matrix[i, topic_id] = prob

    # Calculate entropy for each sentence
    entropy = -np.sum(topic_matrix * np.log2(topic_matrix + 1e-10), axis=1)

    # Calculate overall topic entropy
    topic_entropy = np.mean(entropy)

    # Calculate variance in topic distribution
    lda_variance = np.var(topic_matrix)

    return topic_matrix, topic_entropy, lda_variance, lda_model


# TextRank Similarity Matrix
def create_similarity_matrix(processed_sentences):
    """
    Create similarity matrix using TextRank methodology
    """
    n = len(processed_sentences)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate similarity using word overlap as per TextRank formula in the paper
                words_i = set(processed_sentences[i])
                words_j = set(processed_sentences[j])

                if len(words_i) == 0 or len(words_j) == 0:
                    continue

                # Implementing the similarity formula from the paper
                overlap = len(words_i.intersection(words_j))
                denominator = math.log(len(words_i) + 1) + math.log(len(words_j) + 1)
                if denominator != 0:
                    similarity_matrix[i, j] = overlap / denominator

    # Row normalization
    for i in range(n):
        row_sum = np.sum(similarity_matrix[i, :])
        if row_sum != 0:
            similarity_matrix[i, :] = similarity_matrix[i, :] / row_sum

    return similarity_matrix


# Calculate Dynamic Damping Factor
def calculate_dynamic_damping_factor(
    similarity_matrix, topic_matrix, topic_entropy, lda_variance, num_topics
):
    """
    Calculate dynamic damping factor based on the paper's formula
    """
    # Calculate Node Connectivity
    n = similarity_matrix.shape[0]
    node_connectivity = np.sum(similarity_matrix > 0) / (n * (n - 1)) if n > 1 else 0
    connectivity_factor = node_connectivity / 5

    # Entropy Adjustment
    entropy_adjust = topic_entropy / 5

    # Length Factor
    length_factor = 0.5  # This is a placeholder - adjust based on document length

    # Topic Factor
    topic_factor = 50 / num_topics if num_topics > 0 else 0

    # Variance Factor
    textrank_variance = np.var(similarity_matrix)
    variance_factor = (lda_variance + textrank_variance) / 2

    # Calculate dynamic damping factor
    damping_factor = (
        0.5
        + 0.25 * connectivity_factor
        - 0.35 * entropy_adjust
        - 0.2 * length_factor
        - 0.3 * topic_factor
        + 0.15 * variance_factor
    )

    # Ensure damping factor is between 0 and 1
    damping_factor = max(0.1, min(0.9, damping_factor))

    return damping_factor


# Apply PageRank with Dynamic Damping Factor
def apply_pagerank(similarity_matrix, damping_factor, max_iter=100, tol=1e-6):
    """
    Apply PageRank algorithm with dynamic damping factor
    """
    n = similarity_matrix.shape[0]
    ranks = np.ones(n) / n  # Initialize ranks uniformly

    for _ in range(max_iter):
        new_ranks = (1 - damping_factor) / n + damping_factor * (
            similarity_matrix.T @ ranks
        )

        # Check convergence
        if np.sum(np.abs(new_ranks - ranks)) < tol:
            break

        ranks = new_ranks

    return ranks


# Integrate LDA and TextRank
def integrate_lda_textrank(
    topic_matrix, textrank_scores, lda_weight=0.3, textrank_weight=0.7
):
    """
    Integrate LDA topic scores with TextRank scores
    """
    # Calculate LDA scores (sum of topic probabilities for each sentence)
    lda_scores = np.sum(topic_matrix, axis=1)

    # Normalize LDA scores
    lda_scores = (
        lda_scores / np.sum(lda_scores) if np.sum(lda_scores) > 0 else lda_scores
    )

    # Combine scores
    combined_scores = lda_weight * lda_scores + textrank_weight * textrank_scores

    return combined_scores


# Main Summarization Function
def improved_textrank_summarize(text, num_topics=5, num_sentences=3, lda_weight=0.3):
    """
    Generate summary using improved TextRank with dynamic damping factor
    """
    # 1. Preprocess the text
    raw_sentences, processed_sentences = preprocess_text(text)

    if len(raw_sentences) <= num_sentences:
        return " ".join(raw_sentences)

    # 2. Apply LDA for topic modeling
    topic_matrix, topic_entropy, lda_variance, lda_model = apply_lda(
        processed_sentences, num_topics
    )

    # 3. Create similarity matrix using TextRank
    similarity_matrix = create_similarity_matrix(processed_sentences)

    # 4. Calculate dynamic damping factor
    damping_factor = calculate_dynamic_damping_factor(
        similarity_matrix, topic_matrix, topic_entropy, lda_variance, num_topics
    )
    print(f"Dynamic damping factor: {damping_factor}")

    # 5. Apply PageRank with dynamic damping factor
    textrank_scores = apply_pagerank(similarity_matrix, damping_factor)

    # 6. Integrate LDA and TextRank scores
    textrank_weight = 1 - lda_weight
    combined_scores = integrate_lda_textrank(
        topic_matrix, textrank_scores, lda_weight, textrank_weight
    )

    # 7. Select top sentences
    top_indices = combined_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)  # Sort to maintain original order

    summary = " ".join([raw_sentences[i] for i in top_indices])

    return summary


# ROUGE evaluation
def evaluate_rouge(predicted_summary, reference_summary):
    """
    Evaluate summary using ROUGE metrics
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, predicted_summary)
    return scores


# Batch evaluation
def evaluate_batch(df, num_samples=100, num_topics=5, num_sentences=3, lda_weight=0.3):
    """
    Evaluate the summarizer on a batch of articles
    """
    if len(df) < num_samples:
        num_samples = len(df)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for i in range(num_samples):
        article = df["article"].iloc[i]
        reference = df["highlights"].iloc[i]

        # Generate summary
        summary = improved_textrank_summarize(
            article,
            num_topics=num_topics,
            num_sentences=num_sentences,
            lda_weight=lda_weight,
        )

        # Evaluate
        scores = evaluate_rouge(summary, reference)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    # Calculate average scores
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)

    return {"rouge1": avg_rouge1, "rouge2": avg_rouge2, "rougeL": avg_rougeL}


# Function to tune hyperparameters
def tune_hyperparameters(df, num_samples=50):
    """
    Tune hyperparameters for the summarizer
    """
    best_rouge = 0
    best_params = {}

    # Grid search
    for num_topics in [3, 5, 7, 10]:
        for lda_weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(f"Testing: num_topics={num_topics}, lda_weight={lda_weight}")

            results = evaluate_batch(
                df,
                num_samples=num_samples,
                num_topics=num_topics,
                lda_weight=lda_weight,
            )

            # Use ROUGE-L as the primary metric
            if results["rougeL"] > best_rouge:
                best_rouge = results["rougeL"]
                best_params = {"num_topics": num_topics, "lda_weight": lda_weight}

            print(f"Results: {results}")

    return best_params, best_rouge


# Main execution
if __name__ == "__main__":
    # Load a small sample of the datasets for faster testing
    train_df, val_df, test_df = load_datasets(
        # limit_train=100, limit_val=50, limit_test=50
    )

    # Example usage
    sample_article = train_df["article"].iloc[0]
    reference_summary = train_df["highlights"].iloc[0]

    # Generate summary
    print("Generating summary...")
    summary = improved_textrank_summarize(
        sample_article, num_topics=5, num_sentences=3, lda_weight=0.3
    )

    print("\nOriginal Article Preview:")
    print(sample_article[:500] + "...")

    print("\nGenerated Summary:")
    print(summary)

    print("\nReference Summary:")
    print(reference_summary)

    # Evaluate
    scores = evaluate_rouge(summary, reference_summary)
    print("\nROUGE Scores:")
    print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_batch(
        test_df,
        num_samples=50,
        num_topics=5,
        num_sentences=3
    )

    print("Final Test Results:")
    print(f"ROUGE-1: {test_results['rouge1']:.4f}")
    print(f"ROUGE-2: {test_results['rouge2']:.4f}")
    print(f"ROUGE-L: {test_results['rougeL']:.4f}")
