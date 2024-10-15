import re
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import os
# from memory_profiler import memory_usage

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Get rid of special characters and numbers
    text = re.sub(r'[^ا-یA-Za-z\s]', '', text)  # Let Persian characters through
    # Break the text into words
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

with open('captions.txt', 'r', encoding='utf-8') as f:
    captions = f.readlines()

# Clean up the captions by removing extra spaces and empty lines
captions = [caption.strip() for caption in captions if caption.strip()]

# Drop any empty captions and put them in a DataFrame
df = pd.DataFrame(captions, columns=['Caption'])
df = df[df['Caption'].str.strip() != '']  # Make sure no blanks sneak in

# Preprocess the captions (clean them up)
df['Processed_Caption'] = df['Caption'].apply(preprocess_text)

# Check if there's anything left after cleaning up
if df.empty:
    print("No valid captions found in the dataset.")
else:
    # Set up the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Convert the cleaned captions into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Caption'])

    # Calculate the similarity between all pairs of captions
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Grab only the upper triangle of the matrix to avoid duplicates
    similarity_scores = cosine_sim[np.triu_indices(len(cosine_sim), k=1)]

    # A function to count how many pairs are similar based on a threshold
    def calculate_percentage(threshold):
        similar_pairs_count = np.sum(similarity_scores >= threshold / 100)
        total_pairs = len(similarity_scores)
        return (similar_pairs_count / total_pairs) * 100 if total_pairs > 0 else 0, similar_pairs_count
    
    # Save the similar pairs for each threshold (10% to 100%) into different files
    os.makedirs('result', exist_ok=True)

    for threshold in range(10, 101, 10):
        with open(f'result/similar_pairs_{threshold}%.txt', 'w', encoding='utf-8') as f:
            for i in range(cosine_sim.shape[0]):
                for j in range(i + 1, cosine_sim.shape[1]):
                    if cosine_sim[i, j] >= threshold / 100:  # If similarity meets the threshold
                        f.write(f"Caption {i} and Caption {j} are {cosine_sim[i, j] * 100:.2f}% similar\n")
                        f.write(f"Caption {i}: {df['Caption'][i]}\n")
                        f.write(f"Caption {j}: {df['Caption'][j]}\n")
                        f.write("\n")
        print(f"Similar pairs for {threshold}% saved to similar_pairs_{threshold}%.txt")
