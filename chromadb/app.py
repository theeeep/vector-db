import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Get or create a collection
collection = chroma_client.get_or_create_collection(name="test")

# Define text documents
documents = [
    {"id": "1", "text": "hello world"},
    {"id": "2", "text": "I'm going to win again"},
    {"id": "3", "text": "First rule of fight club"},
    {"id": "4", "text": "You shouldn't discuss about fight club"},
    {"id": "5", "text": "Second rule of fight club"},
]

# Upsert documents into the collection
for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

# Define a query text
query_text = "discuss"

# Query the collection
results = collection.query(query_texts=[query_text], n_results=3)

# Process and print the results
for i in range(len(results["ids"][0])):  # Iterate over the results for the first query
    doc_id = results["ids"][0][i]  # Get the document ID
    distance = results["distances"][0][i]  # Get the distance
    document_text = results["documents"][0][i]  # Get the document text
    print(
        f"For the query: {query_text}, the document {doc_id} has a distance of {distance} and text: {document_text}"
    )
