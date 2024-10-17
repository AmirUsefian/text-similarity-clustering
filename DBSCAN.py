import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class CaptionSimilarity:
    def __init__(self, captions_file):
        self.captions_file = captions_file
        self.captions = self.load_captions()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.df = pd.DataFrame(self.captions, columns=['Caption'])
        self.df['Processed_Caption'] = self.df['Caption'].apply(self.preprocess_text)
        self.cosine_sim = None  # Cosine similarity matrix

    def load_captions(self):
        """Load captions from the provided file and clean empty lines."""
        try:
            with open(self.captions_file, 'r', encoding='utf-8') as f:
                captions = f.readlines()
            captions = [caption.strip() for caption in captions if caption.strip()]
            if not captions:
                raise ValueError("The captions file is empty or all lines are invalid.")
            print(f"{len(captions)} valid captions loaded from {self.captions_file}.")
            return captions
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.captions_file} not found.")
        except Exception as e:
            raise Exception(f"Error reading the captions file: {str(e)}")

    def preprocess_text(self, text):
        """Clean text by removing special characters, numbers, and stopwords; lemmatize words."""
        try:
            text = re.sub(r'[^ا-یA-Za-z\s]', '', text)  # Remove non-Persian/English characters
            words = text.lower().split()
            words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
            return ' '.join(words)
        except Exception as e:
            raise Exception(f"Error processing text: {str(e)}")

    def calculate_similarity(self):
        """Calculate cosine similarity between the captions using TF-IDF vectorization."""
        try:
            if self.df.empty:
                raise ValueError("No valid captions available after preprocessing.")

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['Processed_Caption'])

            # Compute cosine similarity matrix
            self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            print("Cosine similarity matrix calculated.")
        except ValueError as e:
            raise ValueError(f"Error in calculating similarity: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

    def save_similar_pairs(self, threshold=50):
        """Save caption pairs that exceed the given similarity threshold to a file."""
        if self.cosine_sim is None:
            raise ValueError("Cosine similarity matrix not computed yet. Call calculate_similarity() first.")

        # Create results folder if it doesn't exist
        os.makedirs('result', exist_ok=True)

        with open(f'result/similar_pairs_{threshold}%.txt', 'w', encoding='utf-8') as f:
            similar_count = 0
            for i in range(self.cosine_sim.shape[0]):
                for j in range(i + 1, self.cosine_sim.shape[1]):
                    if self.cosine_sim[i, j] >= threshold / 100:
                        f.write(f"Caption {i} and Caption {j} are {self.cosine_sim[i, j] * 100:.2f}% similar\n")
                        f.write(f"Caption {i}: {self.df['Caption'][i]}\n")
                        f.write(f"Caption {j}: {self.df['Caption'][j]}\n\n")
                        similar_count += 1
            print(f"{similar_count} similar pairs (>= {threshold}%) saved to result/similar_pairs_{threshold}%.txt")

    def process(self, thresholds=None):
        """Main processing function that calculates similarity and saves results for given thresholds."""
        if thresholds is None:
            thresholds = range(10, 101, 10)  # Default to thresholds from 10% to 100%

        self.calculate_similarity()

        # Save pairs for each threshold
        for threshold in thresholds:
            print(f"Processing for {threshold}% similarity threshold...")
            self.save_similar_pairs(threshold)

# Example usage:
if __name__ == "__main__":
    captions_processor = CaptionSimilarity('captions.txt')
    captions_processor.process([30, 50, 70])  # Process and save similar pairs for 30%, 50%, and 70% thresholds
