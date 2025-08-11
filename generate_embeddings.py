import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_DIMENSION = 384 

def chunk_text(text, max_chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for better embedding coverage.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_chunk_size - overlap):
        chunk = ' '.join(words[i:i + max_chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def generate_rental_embeddings():
    """
    Generates embeddings for rental property information from text file.
    """
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Read the rental information text file
    with open("rental_info.txt", "r", encoding='utf-8') as f:
        rental_text = f.read().strip()
    
    # Split text into manageable chunks
    text_chunks = chunk_text(rental_text)
    
    # Generate embeddings for each chunk
    embeddings = []
    for chunk in text_chunks:
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    
    # Save the chunks for later retrieval
    with open("rental_chunks.json", "w", encoding='utf-8') as f:
        json.dump(text_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(embeddings)} embeddings from {len(text_chunks)} text chunks")
    return embeddings

def save_embeddings(embeddings):
    """
    Saves the generated embeddings to a file using FAISS.
    """
    
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    
    faiss.write_index(index, "rental_embeddings.index")
    print("Rental embeddings saved")

if __name__ == "__main__":
    embeddings = generate_rental_embeddings()
    save_embeddings(embeddings)