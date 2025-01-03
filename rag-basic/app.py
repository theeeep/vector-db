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

# *Initalize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma-storage")
collection_name = "document_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

# *Initialize OpenAI
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


# *Function to load document from a directory
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


# *Load documents from the directory
directory_path = "./news_article"
documents = load_docuemnts_from_directory(directory_path)

print(f"---- Loaded {len(documents)} documents ----")

# *Split documents into chunks
chunked_doc = []
for doc in documents:
    chunks = split_text_into_chunks(doc["content"])
    for i, chunk in enumerate(chunks):
        chunked_doc.append({"id": f"{doc['id']}_chunk{i+1}", "content": chunk})

print(f"---- Split {len(documents)} documents into {len(chunks)} chunks ----")


# *Function to generate embeddings using OpenAI API
def get_openai_embeddings(content):
    response = client.embeddings.create(input=content, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print(f"---- Generating embeddings for {content} ----")
    return embedding


# *Generate embeddings for the document chunks
for doc in chunked_doc:
    print(f"---- Generating embeddings for {doc['id']} ----")
    doc["embedding"] = get_openai_embeddings(doc["content"])
    # print(doc["embedding"])

# *Upsert documents into ChromaDB
for doc in chunked_doc:
    print(f"---- Upserting {len(chunked_doc)} documents into ChromaDB ----")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["content"]],
        embeddings=[doc["embedding"]],
    )
# collection.upsert(
#     documents=chunked_doc,
#     ids=[doc["id"] for doc in chunked_doc],
#     embeddings=[doc["embedding"] for doc in chunked_doc],
# )


# *Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("---- Returning relevant chunks ----")
    return relevant_chunks


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
