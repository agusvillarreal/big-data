import numpy as np
import re
import math
from collections import Counter
from typing import List, Dict, Tuple


class VectorSpaceModel:
    """
    Implementation of the Vector Space Model for Information Retrieval
    with TF-IDF weighting and cosine similarity.
    """
    
    def __init__(self):
        self.documents = []
        self.document_tokens = []
        self.vocabulary = set()
        self.idf = {}
        self.tfidf_matrix = None
        self.document_norms = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase, removing special characters,
        and tokenizing into words.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            A list of tokens (words)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and replace with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize (split into words)
        tokens = text.split()
        
        # Optional: Remove stop words (common words like 'the', 'a', 'and', etc.)
        # For a real implementation, you might want to use a predefined list of stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def add_document(self, doc_text: str) -> None:
        """
        Add a document to the corpus.
        
        Args:
            doc_text: The text of the document to add
        """
        self.documents.append(doc_text)
        tokens = self.preprocess_text(doc_text)
        self.document_tokens.append(tokens)
        self.vocabulary.update(tokens)
    
    def build_tfidf_matrix(self) -> None:
        """
        Compute the TF-IDF matrix for the document corpus.
        Each row represents a document, each column represents a term in the vocabulary.
        """
        n_documents = len(self.documents)
        vocabulary_list = sorted(list(self.vocabulary))
        vocab_size = len(vocabulary_list)
        
        # Create a dictionary to map terms to indices
        term_to_idx = {term: idx for idx, term in enumerate(vocabulary_list)}
        
        # Initialize TF-IDF matrix
        self.tfidf_matrix = np.zeros((n_documents, vocab_size))
        
        # Calculate document frequencies (DF) for each term
        df = {}
        for doc_tokens in self.document_tokens:
            # Get unique terms in this document
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
        
        # Calculate IDF for each term
        self.idf = {}
        for term, document_freq in df.items():
            self.idf[term] = math.log(n_documents / document_freq)
        
        # Compute TF-IDF values
        for doc_idx, doc_tokens in enumerate(self.document_tokens):
            # Count term frequencies in this document
            term_counts = Counter(doc_tokens)
            doc_length = len(doc_tokens)
            
            # Calculate TF-IDF for each term in this document
            for term, count in term_counts.items():
                if term in term_to_idx:  # Check if term is in vocabulary
                    term_idx = term_to_idx[term]
                    tf = count / doc_length  # Normalize by document length
                    self.tfidf_matrix[doc_idx, term_idx] = tf * self.idf[term]
        
        # Precompute document vector norms for cosine similarity
        self.document_norms = np.sqrt(np.sum(self.tfidf_matrix ** 2, axis=1))
    
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Process a query and return the top-k most relevant documents.
        
        Args:
            query_text: The query text
            top_k: Number of top documents to return
            
        Returns:
            A list of tuples (document_index, similarity_score) sorted by relevance
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix has not been built. Call build_tfidf_matrix() first.")
        
        # Preprocess the query
        query_tokens = self.preprocess_text(query_text)
        
        # Create a query vector
        vocabulary_list = sorted(list(self.vocabulary))
        term_to_idx = {term: idx for idx, term in enumerate(vocabulary_list)}
        query_vector = np.zeros(len(vocabulary_list))
        
        # Count term frequencies in the query
        term_counts = Counter(query_tokens)
        query_length = len(query_tokens)
        
        # Compute TF-IDF for the query vector
        for term, count in term_counts.items():
            if term in term_to_idx and term in self.idf:  # Ensure term is in vocabulary and IDF
                term_idx = term_to_idx[term]
                tf = count / query_length
                query_vector[term_idx] = tf * self.idf.get(term, 0)  # Use 0 if term not in corpus
        
        # Compute query vector norm
        query_norm = np.sqrt(np.sum(query_vector ** 2))
        if query_norm == 0:  # Handle zero query vector
            return []
        
        # Calculate cosine similarity between query and all documents
        cosine_similarities = np.zeros(len(self.documents))
        for i in range(len(self.documents)):
            if self.document_norms[i] == 0:  # Handle zero document vector
                cosine_similarities[i] = 0
            else:
                dot_product = np.dot(self.tfidf_matrix[i], query_vector)
                cosine_similarities[i] = dot_product / (self.document_norms[i] * query_norm)
        
        # Get indices of top-k documents
        top_indices = np.argsort(-cosine_similarities)[:top_k]
        
        # Return document indices and their similarity scores
        return [(idx, cosine_similarities[idx]) for idx in top_indices if cosine_similarities[idx] > 0]
    
    def explain_query(self, query_text: str, doc_idx: int) -> Dict:
        """
        Explain why a document is relevant to a query by showing term contributions.
        
        Args:
            query_text: The query text
            doc_idx: The index of the document to explain
            
        Returns:
            A dictionary with explanation details
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix has not been built. Call build_tfidf_matrix() first.")
        
        # Preprocess the query
        query_tokens = self.preprocess_text(query_text)
        
        # Create a query vector with explanations
        vocabulary_list = sorted(list(self.vocabulary))
        term_to_idx = {term: idx for idx, term in enumerate(vocabulary_list)}
        
        # Prepare explanation data
        term_contributions = {}
        query_length = len(query_tokens)
        
        # Compute and store term contributions
        for term in set(query_tokens):
            if term in term_to_idx and term in self.idf:
                term_idx = term_to_idx[term]
                query_tf = query_tokens.count(term) / query_length
                query_tfidf = query_tf * self.idf.get(term, 0)
                doc_tfidf = self.tfidf_matrix[doc_idx, term_idx]
                
                # Contribution to similarity is the product of term weights in query and document
                contribution = query_tfidf * doc_tfidf
                
                term_contributions[term] = {
                    'term': term,
                    'query_tf': query_tf,
                    'query_tfidf': query_tfidf,
                    'doc_tfidf': doc_tfidf,
                    'contribution': contribution
                }
        
        # Sort terms by contribution
        sorted_terms = sorted(term_contributions.keys(), 
                             key=lambda t: term_contributions[t]['contribution'], 
                             reverse=True)
        
        # Final explanation
        explanation = {
            'document': self.documents[doc_idx],
            'query': query_text,
            'cosine_similarity': self.query(query_text, top_k=len(self.documents))[0][1] if doc_idx == self.query(query_text, top_k=1)[0][0] else None,
            'term_contributions': [term_contributions[term] for term in sorted_terms]
        }
        
        return explanation


# Example usage
def main():
    # Create a sample corpus
    corpus = [
        "Big data processing requires distributed computing frameworks.",
        "Hadoop and Spark are popular big data processing tools.",
        "Information retrieval systems help find relevant documents.",
        "Search engines use inverted indices for efficient document retrieval.",
        "TF-IDF and VSM are fundamental concepts in information retrieval.",
        "Vector space models represent documents as vectors in a term space.",
        "Machine learning algorithms can improve search relevance."
    ]
    
    # Initialize the VSM
    vsm = VectorSpaceModel()
    
    # Add documents to the corpus
    for doc in corpus:
        vsm.add_document(doc)
    
    # Build the TF-IDF matrix
    vsm.build_tfidf_matrix()
    
    # Example query
    query = "big data processing with Spark"
    print(f"Query: '{query}'\n")
    
    # Get top 3 relevant documents
    results = vsm.query(query, top_k=3)
    
    print("Top 3 relevant documents:")
    for idx, sim_score in results:
        print(f"Document {idx}: (Score: {sim_score:.4f})")
        print(f"Text: '{corpus[idx]}'\n")
    
    # Explain the top result
    if results:
        top_doc_idx = results[0][0]
        explanation = vsm.explain_query(query, top_doc_idx)
        
        print(f"Explanation for Document {top_doc_idx}:")
        print(f"Cosine similarity: {explanation['cosine_similarity']:.4f}\n")
        
        print("Term contributions:")
        for term_info in explanation['term_contributions']:
            term = term_info['term']
            contrib = term_info['contribution']
            print(f"- '{term}': {contrib:.4f}")


if __name__ == "__main__":
    main()
