import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# ef -> embedding function of OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small"
)

# Initalize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma-storage")
collection_name = "document_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

# Initialize OpenAI
client = OpenAI(api_key=openai_api_key)

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What developer should focus on 2025 like the most?",
#         },
#     ],
# )
# print(response.choices[0].message.content)


# Function to load document from a directory
def load_docuemnts_from_directory(directory_path):
    print(f"---- Loading documents from {directory_path} ----")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r") as file:
                documents.append({"id": filename, "content": file.read()})
    return documents


# Function to split text into chunks
# text: The input string that you want to split into chunks.
# chunk_size: The maximum size of each chunk (default is 1000 characters).
# overlap: The number of characters that should overlap between consecutive chunks (default is 20 characters).
# Example:
# If you have a text of 3000 characters and you set chunk_size to 1000 and overlap to 20, the function will produce 3 chunks:
# Chunk 1: Characters 0 to 1000
# Chunk 2: Characters 980 to 1980 (20 characters overlap)
# Chunk 3: Characters 1960 to 2960 (20 characters overlap)
def split_text_into_chunks(text, chunk_size=1000, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# Load documents from the directory
directory_path = "./news_article"
documents = load_docuemnts_from_directory(directory_path)

print(f"---- Loaded {len(documents)} documents ----")

# Split documents into chunks
for doc in documents:
    chunks = split_text_into_chunks(doc["content"])
    for i, chunk in enumerate(chunks):
        chunked_doc = {"id": f"{doc['id']}_{i}", "content": chunk}

print(f"---- Split {len(documents)} documents into {len(chunks)} chunks ----")
