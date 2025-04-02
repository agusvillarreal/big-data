# Vector Space Model (VSM) for Information Retrieval
# Interactive Demo Notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from collections import Counter
from typing import List, Dict, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Set up the styling
plt.style.use('ggplot')
sns.set(font_scale=1.2)


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
        self.vocabulary_list = []
        self.term_to_idx = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase, removing special characters,
        and tokenizing into words.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and replace with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize (split into words)
        tokens = text.split()
        
        # Optional: Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def add_document(self, doc_text: str) -> None:
        """Add a document to the corpus."""
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
        self.vocabulary_list = sorted(list(self.vocabulary))
        vocab_size = len(self.vocabulary_list)
        
        # Create a dictionary to map terms to indices
        self.term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary_list)}
        
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
                term_idx = self.term_to_idx[term]
                tf = count / doc_length  # Normalize by document length
                self.tfidf_matrix[doc_idx, term_idx] = tf * self.idf[term]
        
        # Precompute document vector norms for cosine similarity
        self.document_norms = np.sqrt(np.sum(self.tfidf_matrix ** 2, axis=1))
    
    def compute_query_vector(self, query_text: str) -> np.ndarray:
        """Compute the TF-IDF vector for a query."""
        query_tokens = self.preprocess_text(query_text)
        query_vector = np.zeros(len(self.vocabulary_list))
        
        # Count term frequencies in the query
        term_counts = Counter(query_tokens)
        query_length = len(query_tokens)
        
        # Compute TF-IDF for the query vector
        for term, count in term_counts.items():
            if term in self.term_to_idx and term in self.idf:
                term_idx = self.term_to_idx[term]
                tf = count / query_length
                query_vector[term_idx] = tf * self.idf.get(term, 0)
                
        return query_vector
        
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Process a query and return the top-k most relevant documents.
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix has not been built. Call build_tfidf_matrix() first.")
        
        # Compute query vector
        query_vector = self.compute_query_vector(query_text)
        
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
    
    def get_term_contributions(self, query_text: str, doc_idx: int) -> Dict:
        """Calculate how much each term contributes to the similarity score."""
        query_tokens = self.preprocess_text(query_text)
        query_vector = self.compute_query_vector(query_text)
        
        contributions = {}
        
        # Calculate the contribution of each term to the similarity
        for term in set(query_tokens).intersection(set(self.document_tokens[doc_idx])):
            if term in self.term_to_idx:
                term_idx = self.term_to_idx[term]
                query_weight = query_vector[term_idx]
                doc_weight = self.tfidf_matrix[doc_idx, term_idx]
                
                # Term contribution to similarity
                contribution = query_weight * doc_weight / (self.document_norms[doc_idx] * np.sqrt(np.sum(query_vector ** 2)))
                
                contributions[term] = contribution
                
        # Sort by contribution (highest first)
        return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


# Demo corpus - information retrieval and search engines concepts
corpus = [
    "Big data processing requires distributed computing frameworks for efficient data analysis.",
    "Hadoop and Spark are popular big data processing tools for handling large datasets.",
    "Information retrieval systems help find relevant documents in response to user queries.",
    "Search engines use inverted indices for efficient document retrieval and faster query processing.",
    "TF-IDF weighting and Vector Space Model are fundamental concepts in information retrieval.",
    "Vector space models represent documents as vectors in a high-dimensional term space.",
    "Machine learning algorithms can improve search relevance by learning from user behavior.",
    "Lucene provides a powerful indexing and search library for building search applications.",
    "Elasticsearch and Solr are popular search platforms built on top of Apache Lucene.",
    "Text preprocessing includes tokenization, stemming, and stop word removal to improve retrieval."
]

# Initialize and build the model
vsm = VectorSpaceModel()
for doc in corpus:
    vsm.add_document(doc)
vsm.build_tfidf_matrix()

# Create a function to display TF-IDF matrix as a heatmap
def plot_tfidf_heatmap(vsm, focus_terms=None):
    """
    Plot the TF-IDF matrix as a heatmap.
    
    Args:
        vsm: Vector Space Model instance
        focus_terms: Optional list of terms to highlight in the plot
    """
    if focus_terms:
        # Get only the columns for the focus terms
        term_indices = [vsm.term_to_idx[term] for term in focus_terms if term in vsm.term_to_idx]
        matrix = vsm.tfidf_matrix[:, term_indices]
        terms = [vsm.vocabulary_list[idx] for idx in term_indices]
    else:
        # Use the entire matrix (could be large!)
        matrix = vsm.tfidf_matrix
        terms = vsm.vocabulary_list
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(matrix, 
                     columns=terms,
                     index=[f"Doc {i}" for i in range(len(vsm.documents))])
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("TF-IDF Matrix Heatmap")
    plt.ylabel("Documents")
    plt.xlabel("Terms")
    plt.tight_layout()
    plt.show()

# Function to visualize document and query vectors
def plot_vectors_2d(vsm, query_text=None, pca=True):
    """
    Plot document vectors and optionally a query vector in 2D space.
    Uses PCA to reduce dimensions for visualization.
    
    Args:
        vsm: Vector Space Model instance
        query_text: Optional query text to include in the plot
        pca: Whether to use PCA (True) or take first 2 dimensions (False)
    """
    from sklearn.decomposition import PCA
    
    # Prepare the document vectors
    vectors = vsm.tfidf_matrix
    
    # Add query vector if provided
    query_vector = None
    if query_text:
        query_vector = vsm.compute_query_vector(query_text)
        vectors_with_query = np.vstack([vectors, query_vector])
    else:
        vectors_with_query = vectors
    
    # Reduce to 2D for visualization
    if pca:
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_with_query)
    else:
        # Just take the first two dimensions (for demonstration)
        vectors_2d = vectors_with_query[:, :2]
    
    # Plot documents
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:len(vectors), 0], vectors_2d[:len(vectors), 1], 
                c='blue', label='Documents', alpha=0.7)
    
    # Add document labels
    for i in range(len(vectors)):
        plt.annotate(f"Doc {i}", (vectors_2d[i, 0], vectors_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot query if provided
    if query_text:
        plt.scatter(vectors_2d[-1, 0], vectors_2d[-1, 1], 
                    c='red', s=100, label='Query', marker='*')
        plt.annotate(f"Query: '{query_text}'", (vectors_2d[-1, 0], vectors_2d[-1, 1]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Document and Query Vectors in 2D Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to visualize cosine similarity
def plot_cosine_similarity(vsm, query_text):
    """
    Visualize the cosine similarity between a query and all documents.
    
    Args:
        vsm: Vector Space Model instance
        query_text: Query text
    """
    # Compute the query vector
    query_vector = vsm.compute_query_vector(query_text)
    query_norm = np.sqrt(np.sum(query_vector ** 2))
    
    if query_norm == 0:
        print("Query vector is empty (no terms match the vocabulary).")
        return
    
    # Calculate cosine similarity for all documents
    similarities = []
    for i in range(len(vsm.documents)):
        if vsm.document_norms[i] == 0:
            similarities.append(0)
        else:
            dot_product = np.dot(vsm.tfidf_matrix[i], query_vector)
            similarities.append(dot_product / (vsm.document_norms[i] * query_norm))
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Document': [f"Doc {i}" for i in range(len(vsm.documents))],
        'Cosine Similarity': similarities
    })
    
    # Sort by similarity (highest first)
    df = df.sort_values('Cosine Similarity', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Cosine Similarity', y='Document', data=df, palette='viridis')
    
    # Add values on bars
    for i, v in enumerate(df['Cosine Similarity']):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    plt.title(f"Cosine Similarity for Query: '{query_text}'")
    plt.xlim(0, 1)  # Cosine similarity is between 0 and 1 for TF-IDF vectors
    plt.tight_layout()
    plt.show()
    
    # Show term contributions for top document
    top_doc_idx = int(df.iloc[0]['Document'].split()[1])
    contributions = vsm.get_term_contributions(query_text, top_doc_idx)
    
    print(f"\nTop document (Doc {top_doc_idx}): {vsm.documents[top_doc_idx]}")
    print("\nTerm contributions to similarity:")
    
    # Plot term contributions
    if contributions:
        plt.figure(figsize=(10, 5))
        terms = list(contributions.keys())
        values = list(contributions.values())
        
        sns.barplot(x=values, y=terms, palette='viridis')
        plt.title(f"Term Contributions to Similarity for Doc {top_doc_idx}")
        plt.xlabel("Contribution")
        plt.ylabel("Term")
        plt.tight_layout()
        plt.show()
    else:
        print("No matching terms between query and document.")

# Create interactive widgets for the demo
def interactive_demo():
    """Create an interactive demo with widgets."""
    
    # Query input
    query_input = widgets.Text(
        value='search engines information retrieval',
        description='Query:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Buttons for different visualizations
    results_button = widgets.Button(description="Show Search Results")
    tfidf_button = widgets.Button(description="Show TF-IDF Matrix")
    vectors_button = widgets.Button(description="Visualize Vectors")
    similarity_button = widgets.Button(description="Cosine Similarity")
    
    # Functions for button clicks
    def on_results_button_clicked(b):
        clear_output(wait=True)
        display(query_input, widgets.HBox([results_button, tfidf_button, vectors_button, similarity_button]))
        
        query = query_input.value
        results = vsm.query(query, top_k=5)
        
        print(f"Top results for query: '{query}'\n")
        if not results:
            print("No relevant documents found.")
            return
            
        for i, (doc_idx, score) in enumerate(results):
            print(f"{i+1}. Document {doc_idx} (Score: {score:.4f})")
            print(f"   {vsm.documents[doc_idx]}\n")
    
    def on_tfidf_button_clicked(b):
        clear_output(wait=True)
        display(query_input, widgets.HBox([results_button, tfidf_button, vectors_button, similarity_button]))
        
        # Only show TF-IDF for terms in the query to avoid huge matrices
        query_terms = set(vsm.preprocess_text(query_input.value))
        plot_tfidf_heatmap(vsm, focus_terms=query_terms)
    
    def on_vectors_button_clicked(b):
        clear_output(wait=True)
        display(query_input, widgets.HBox([results_button, tfidf_button, vectors_button, similarity_button]))
        
        plot_vectors_2d(vsm, query_text=query_input.value)
    
    def on_similarity_button_clicked(b):
        clear_output(wait=True)
        display(query_input, widgets.HBox([results_button, tfidf_button, vectors_button, similarity_button]))
        
        plot_cosine_similarity(vsm, query_input.value)
    
    # Attach click handlers
    results_button.on_click(on_results_button_clicked)
    tfidf_button.on_click(on_tfidf_button_clicked)
    vectors_button.on_click(on_vectors_button_clicked)
    similarity_button.on_click(on_similarity_button_clicked)
    
    # Display the widgets
    display(HTML("<h1>Vector Space Model Interactive Demo</h1>"))
    display(query_input)
    display(widgets.HBox([results_button, tfidf_button, vectors_button, similarity_button]))
    
    # Show initial results
    on_results_button_clicked(None)

# Call the interactive demo function
interactive_demo()
