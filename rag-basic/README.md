# **Project: Document Embedding and Query System**

This project is designed to process, store, and query text documents efficiently using **OpenAI's embeddings** and **ChromaDB** for vector storage. It allows users to load documents, split them into manageable chunks, generate embeddings, store them in a vector database, and query the database to retrieve relevant information. The system also integrates **OpenAI's GPT model** to generate concise and accurate responses based on the retrieved data.

---

## **Key Features**
1. **Document Loading**: Load `.txt` files from a specified directory.
2. **Text Chunking**: Split large documents into smaller, overlapping chunks for better processing.
3. **Embedding Generation**: Use OpenAI's `text-embedding-3-small` model to generate embeddings for each chunk.
4. **Vector Storage**: Store document chunks and their embeddings in **ChromaDB** for efficient retrieval.
5. **Query System**: Query the database to retrieve the most relevant document chunks based on a user's question.
6. **Response Generation**: Use OpenAI's GPT model to generate concise and accurate answers based on the retrieved chunks.

---

## **Use Cases**
- **Document Retrieval**: Quickly find relevant information from a large collection of documents.
- **Question Answering**: Generate accurate answers to user questions based on stored documents.
- **Knowledge Management**: Organize and query large volumes of text data efficiently.

---

Let me know if you'd like to expand or modify this description! 😊

### **Step-by-Step Workflow**

1. **Load Environment Variables**
   - **Function**: `load_dotenv()`
   - **What Happens**: Loads the OpenAI API key from the `.env` file.

2. **Initialize OpenAI Embedding Function**
   - **Function**: `embedding_functions.OpenAIEmbeddingFunction()`
   - **What Happens**: Sets up the embedding function using OpenAI's API.

3. **Initialize ChromaDB Client and Collection**
   - **Functions**:
     - `chromadb.PersistentClient()`
     - `chroma_client.get_or_create_collection()`
   - **What Happens**: Creates or connects to a ChromaDB collection for storing document embeddings.

4. **Initialize OpenAI Client**
   - **Function**: `OpenAI()`
   - **What Happens**: Sets up the OpenAI client for generating embeddings and responses.

5. **Load Documents from Directory**
   - **Function**: `load_docuemnts_from_directory()`
   - **What Happens**: Loads `.txt` files from a specified directory and returns a list of documents with their IDs and content.

6. **Split Documents into Chunks**
   - **Function**: `split_text_into_chunks()`
   - **What Happens**: Splits the content of each document into smaller chunks for processing.

7. **Generate Embeddings for Document Chunks**
   - **Function**: `get_openai_embeddings()`
   - **What Happens**: Generates embeddings for each chunk using OpenAI's embedding model.

8. **Upsert Documents into ChromaDB**
   - **Function**: `collection.upsert()`
   - **What Happens**: Inserts or updates the document chunks and their embeddings into the ChromaDB collection.

9. **Query Documents**
   - **Function**: `query_documents()`
   - **What Happens**: Queries the ChromaDB collection for relevant document chunks based on a user question.

10. **Generate Response Using OpenAI**
    - **Function**: `generate_response()`
    - **What Happens**: Generates a concise response to the user's question using the retrieved document chunks.

---

### **Functions Overview**

1. **`load_docuemnts_from_directory(directory_path)`**
   - Loads `.txt` files from a directory.

2. **`split_text_into_chunks(text, chunk_size=1000, overlap=20)`**
   - Splits text into chunks with specified size and overlap.

3. **`get_openai_embeddings(content)`**
   - Generates embeddings for text using OpenAI's embedding model.

4. **`query_documents(question, n_results=2)`**
   - Queries ChromaDB for relevant document chunks.

5. **`generate_response(question, relevant_chunks)`**
   - Generates a response using OpenAI's GPT model and retrieved chunks.

---

### **Example Usage**

1. **Load Documents**:
   ```python
   documents = load_docuemnts_from_directory("./news_article")
   ```

2. **Split Documents into Chunks**:
   ```python
   chunked_doc = []
   for doc in documents:
       chunks = split_text_into_chunks(doc["content"])
       for i, chunk in enumerate(chunks):
           chunked_doc.append({"id": f"{doc['id']}_chunk{i+1}", "content": chunk})
   ```

3. **Generate Embeddings and Upsert into ChromaDB**:
   ```python
   for doc in chunked_doc:
       doc["embedding"] = get_openai_embeddings(doc["content"])
       collection.upsert(
           ids=[doc["id"]],
           documents=[doc["content"]],
           embeddings=[doc["embedding"]],
       )
   ```

4. **Query and Generate Response**:
   ```python
   question = "tell me about databricks"
   relevant_chunks = query_documents(question)
   answer = generate_response(question, relevant_chunks)
   print(answer)
   ```

---

### **Dependencies**
- `os`
- `dotenv`
- `chromadb`
- `openai`
- `chromadb.utils.embedding_functions`

---

### **Environment Variables**
- `OPENAI_API_KEY`: Your OpenAI API key stored in a `.env` file.

---