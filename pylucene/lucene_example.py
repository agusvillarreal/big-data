import os
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser

# Initialize JVM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])
print("PyLucene version:", lucene.VERSION)

# Create an index directory
index_dir = "lucene_index"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# Set up the analyzer and index writer
store = FSDirectory.open(Paths.get(index_dir))
analyzer = StandardAnalyzer()
config = IndexWriterConfig(analyzer)
config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
writer = IndexWriter(store, config)

print("Created index writer successfully")

# Function to add documents to the index
def add_document(writer, title, content, author):
    doc = Document()
    doc.add(StringField("title", title, Field.Store.YES))
    doc.add(TextField("content", content, Field.Store.YES))
    doc.add(StringField("author", author, Field.Store.YES))
    writer.addDocument(doc)
    print(f"Added document: {title}")

# Sample documents
sample_docs = [
    {
        "title": "Big Data Processing with Spark",
        "content": "Apache Spark is a unified analytics engine for big data processing, with built-in modules for streaming, SQL, machine learning and graph processing.",
        "author": "Tech Writer"
    },
    {
        "title": "Introduction to Hadoop",
        "content": "Hadoop is a framework that allows for distributed processing of large data sets across clusters of computers using simple programming models.",
        "author": "Data Engineer"
    },
    {
        "title": "Modern Data Lakes",
        "content": "A data lake is a centralized repository that allows you to store all your structured and unstructured data at any scale.",
        "author": "Cloud Architect"
    }
]

# Add documents to the index
for doc in sample_docs:
    add_document(writer, doc["title"], doc["content"], doc["author"])

# Commit changes and close the writer
writer.commit()
writer.close()
print("Indexing completed successfully!")

# Open the index for searching
reader = DirectoryReader.open(store)
searcher = IndexSearcher(reader)
query_parser = QueryParser("content", analyzer)

def search(query_string, field="content", max_hits=10):
    print(f"\nSearching for: '{query_string}' in field '{field}'")
    query = query_parser.parse(query_string)
    hits = searcher.search(query, max_hits)
    
    # Fix for accessing totalHits
    total_hits = hits.totalHits.value
    print(f"Found {total_hits} document(s)")
    
    for i, hit in enumerate(hits.scoreDocs):
        # Updated method to retrieve documents in PyLucene 10.0.0
        doc = searcher.storedFields().document(hit.doc)
        score = hit.score
        print(f"\nResult {i+1}: Score = {score:.4f}")
        print(f"Title: {doc.get('title')}")
        print(f"Author: {doc.get('author')}")
        print(f"Content: {doc.get('content')}")

# Perform some example searches
search("big data")
search("apache")
search("framework")

# Clean up resources
reader.close()
store.close()
print("\nSearch completed and resources cleaned up")