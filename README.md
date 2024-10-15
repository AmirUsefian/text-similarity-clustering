This project is a Python-based implementation designed to analyze the similarity between textual captions using Natural Language Processing (NLP) techniques. The primary objective is to compare captions using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and measure the similarity between them using cosine similarity. Based on these similarity metrics, captions are grouped using DBSCAN clustering.

The code performs the following tasks:

    Cleans and preprocesses the captions, including removing special characters, numbers, and stopwords. It supports both English and Persian text.
    Utilizes TF-IDF Vectorizer to convert the cleaned captions into vectors.
    Computes cosine similarity scores between all pairs of captions.
    Identifies and groups captions that meet a certain similarity threshold.
    Outputs pairs of captions with similarity percentages for various thresholds (10% to 100%).
    Optionally applies clustering algorithms like DBSCAN to group similar captions based on these scores.

Features:

    Text Preprocessing: Removes stopwords, lemmatizes words, and supports basic Persian and English text cleaning.
    TF-IDF Vectorization: Captions are converted into vectors based on their term frequency-inverse document frequency.
    Cosine Similarity: Measures the similarity between pairs of captions, enabling comparison across a range of thresholds.
    Threshold-Based Similarity Output: For each similarity threshold, the program generates a file containing pairs of captions and their similarity score.
    Clustering with DBSCAN: Groups captions based on similarity scores, allowing for analysis of clusters of semantically similar content.
    Memory Profiling: Monitors memory usage to ensure efficient handling of the dataset.

Use Cases:

    Content Moderation: Identify near-duplicate captions or detect plagiarism across text datasets.
    Data Deduplication: Group similar captions together, useful in text-based datasets that require deduplication or organization.
    Textual Analysis: Gain insights into how similar different captions or pieces of text are, with applications in NLP tasks like text summarization or classification.

Requirements:

    Python 3.8+
    Libraries: scikit-learn, pandas, nltk, numpy, memory-profiler
