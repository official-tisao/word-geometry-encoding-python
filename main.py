#!/usr/bin/env python3
"""
Text Geometry Encoder using Gensim Word Embeddings
Converts input text into numerical vectors (geometry) using pre-trained word embeddings
"""

import numpy as np
import gensim.downloader
from typing import List, Optional, Union
import re

class TextGeometryEncoder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        """
        Initialize the text geometry encoder with a pre-trained model
        
        Args:
            model_name: Name of the pre-trained model to use
                       Available options: 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                       'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'word2vec-google-news-300'
        """
        print(f"Loading model: {model_name}")
        self.model = gensim.downloader.load(model_name)
        self.model_name = model_name
        self.vector_size = self.model.vector_size
        if model_name in TextGeometryEncoder._model_cache:
            print(f"Using cached model: {model_name}")
            self.model = TextGeometryEncoder._model_cache[model_name]
        else:
            print(f"Loading model: {model_name}")
            try:
                self.model = gensim.downloader.load(model_name)
                TextGeometryEncoder._model_cache[model_name] = self.model
                print(f"Model loaded successfully. Vector dimension: {self.model.vector_size}")
            except (ValueError, IOError, OSError, MemoryError) as e:
                print(f"Error loading model '{model_name}': {e}")
                raise
            except Exception as e:
                print(f"Unexpected error loading model '{model_name}': {e}")
                raise
        self.model_name = model_name
        self.vector_size = self.model.vector_size
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by cleaning and tokenizing
        
        Args:
            text: Input text string
            
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get vector representation of a single word
        
        Args:
            word: Input word
            
        Returns:
            Vector representation or None if word not in vocabulary
        """
        try:
            return self.model[word]
        except KeyError:
            return None
    
    def encode_text(self, text: str, method: str = "mean") -> np.ndarray:
        """
        Convert text into a single vector representation
        
        Args:
            text: Input text string
            method: Aggregation method - 'mean', 'sum', or 'max'
            
        Returns:
            Vector representation of the text
        """
        tokens = self.preprocess_text(text)
        vectors = []
        
        for token in tokens:
            vector = self.get_word_vector(token)
            if vector is not None:
                vectors.append(vector)
        
        if not vectors:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.vector_size)
        
        vectors = np.array(vectors)
        
        if method == "mean":
            return np.mean(vectors, axis=0)
        elif method == "sum":
            return np.sum(vectors, axis=0)
        elif method == "max":
            return np.max(vectors, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def encode_words_separately(self, text: str) -> dict:
        """
        Get vector representations for each word separately
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary mapping words to their vectors
        """
        tokens = self.preprocess_text(text)
        word_vectors = {}
        
        for token in tokens:
            vector = self.get_word_vector(token)
            if vector is not None:
                word_vectors[token] = vector
            else:
                print(f"Warning: '{token}' not found in vocabulary")
        
        return word_vectors
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[tuple]:
        """
        Find words similar to the input word
        
        Args:
            word: Input word
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            print(f"'{word}' not found in vocabulary")
            return []
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            print(f"Word not found: {e}")
            return 0.0


def main():
    """Example usage of the TextGeometryEncoder"""
    
    # Initialize the encoder
    encoder = TextGeometryEncoder("glove-wiki-gigaword-50")
    
    # Example text inputs
    sample_texts = [
        "artificial intelligence machine learning",
        "the quick brown fox jumps",
        "computer science programming python",
        "natural language processing"
    ]
    
    print("\n" + "="*60)
    print("TEXT GEOMETRY ENCODING EXAMPLES")
    print("="*60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nExample {i}: '{text}'")
        print("-" * 40)
        
        # Get overall text vector
        text_vector = encoder.encode_text(text, method="mean")
        print(f"Text vector shape: {text_vector.shape}")
        print(f"First 10 dimensions: {text_vector[:10]}")
        
        # Get individual word vectors
        word_vectors = encoder.encode_words_separately(text)
        print(f"Individual words encoded: {len(word_vectors)}")
        
        for word, vector in word_vectors.items():
            print(f"  '{word}': shape {vector.shape}")
    
    # Demonstrate similarity calculations
    print(f"\n" + "="*60)
    print("WORD SIMILARITY EXAMPLES")
    print("="*60)
    
    word_pairs = [("king", "queen"), ("computer", "laptop"), ("cat", "dog")]
    
    for word1, word2 in word_pairs:
        similarity = encoder.calculate_similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")
    
    # Find similar words
    print(f"\nWords similar to 'computer':")
    similar = encoder.find_similar_words("computer", top_n=5)
    for word, score in similar:
        print(f"  {word}: {score:.3f}")


if __name__ == "__main__":
    # You can run this as a script
    main()
    
    # Or use it interactively:
    """
    # Example interactive usage:
    encoder = TextGeometryEncoder()
    
    # Encode a single sentence
    my_text = "hello world machine learning"
    vector = encoder.encode_text(my_text)
    print(f"Vector: {vector}")
    
    # Get word vectors separately
    word_vectors = encoder.encode_words_separately(my_text)
    print(f"Word vectors: {word_vectors}")
    """
