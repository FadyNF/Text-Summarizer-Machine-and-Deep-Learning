import numpy as np
from gensim import corpora, models
from main import (
    preprocess_text,
    create_similarity_matrix,
    calculate_dynamic_damping_factor,
    apply_pagerank,
    integrate_lda_textrank,
)

# Define a function to load text from a file
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Specify the file path
file_path = "extended_textrank/sample_text.txt"  # Adjust this to the actual path

# Load the text from the file
sample_text = load_text_from_file(file_path)


def load_lda_model_and_dictionary(
    model_path="extended_textrank/saved_model/lda_model",
    dict_path="extended_textrank/saved_model/lda_dictionary.dict",
):
    lda_model = models.LdaModel.load(model_path)
    dictionary = corpora.Dictionary.load(dict_path)
    return lda_model, dictionary


def improved_textrank_summarize(
    text, lda_model, dictionary, num_topics=5, num_sentences=3, lda_weight=0.3
):
    # 1. Preprocess
    raw_sentences, processed_sentences = preprocess_text(text)
    if len(raw_sentences) <= num_sentences:
        return {
            "summary": " ".join(raw_sentences),
            "topic_entropy": None,
            "lda_variance": None,
        }

    # 2. Convert to BOW using loaded dictionary
    corpus = [dictionary.doc2bow(sent) for sent in processed_sentences]

    # 3. Get topic matrix
    topic_matrix = np.zeros((len(processed_sentences), num_topics))
    for i, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_dist:
            topic_matrix[i, topic_id] = prob

    # 4. Entropy + Variance
    topic_entropy = -np.sum(topic_matrix * np.log2(topic_matrix + 1e-10), axis=1).mean()
    lda_variance = np.var(topic_matrix)

    # 5. TextRank
    similarity_matrix = create_similarity_matrix(processed_sentences)
    damping_factor = calculate_dynamic_damping_factor(
        similarity_matrix, topic_matrix, topic_entropy, lda_variance, num_topics
    )
    textrank_scores = apply_pagerank(similarity_matrix, damping_factor)

    # 6. Combine scores
    combined_scores = integrate_lda_textrank(
        topic_matrix, textrank_scores, lda_weight, 1 - lda_weight
    )

    # 7. Select top sentences
    top_indices = combined_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)

    # Return the summary along with topic entropy and variance
    return {
        "summary": " ".join([raw_sentences[i] for i in top_indices]),
        "topic_entropy": topic_entropy,
        "lda_variance": lda_variance,
    }


# Load the LDA model and dictionary
lda_model, dictionary = load_lda_model_and_dictionary()

# Run the summarization
result = improved_textrank_summarize(
    sample_text,
    lda_model=lda_model,
    dictionary=dictionary,
    num_topics=5,
    num_sentences=5,
    lda_weight=0.3,
)

# Print just the summary cleanly
print("\n🔹 Extractive Summary:\n")
print(result["summary"])