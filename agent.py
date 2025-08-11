import os
import json

import numpy as np
import faiss
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

@st.cache_resource
def load_data():
    """Loads reusable data."""
    
    index = faiss.read_index("rental_embeddings.index")

    with open("rental_chunks.json", "r", encoding='utf-8') as f:
        rental_chunks = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return index, rental_chunks, model

def get_relevant_answers(query, top_k=3):
    """
    Chatbot function that retrieves relevant rental information based on the user's query.
    """

    index, rental_chunks, model = load_data()
    
    query_embedding = model.encode(query)

    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)

    relevant_chunks = []
    for i in range(top_k):
        if indices[0][i] != -1:
            relevant_chunks.append(rental_chunks[indices[0][i]])

    return relevant_chunks

def generate_prompt(query):
    """
    Generates a prompt for the chatbot based on the user's query and the relevant rental information.
    """
    
    relevant_chunks = get_relevant_answers(query)
    
    context = ""
    for chunk in relevant_chunks:
        context += f"Information: {chunk}\n\n"
    
    if not context.strip():
        return "I'm sorry, but I couldn't find any relevant information to answer your question."

    system_prompt = f"""You are a helpful and friendly rental agent chatbot designed to answer questions about rental properties based on the provided information. 
    Your expertise is focused on rental properties, lease terms, amenities, location details, and related rental services.

    **Instructions:**
    1.  Answer the user's question based only on the information provided in the "Context" section below.
    2.  Be concise and provide informative answers.
    3.  If the answer cannot be found within the context, politely refuse to answer by saying: "I'm sorry, but the answer to your question is not covered in the provided rental information."
    4.  If the user asks a question outside the topics of rental properties and related services, politely redirect them by saying: "I can only answer questions related to rental properties and our services."
    5.  When providing information, be specific and helpful for potential tenants.
    6.  Do not invent or assume information not present in the rental information.
    7.  Maintain a friendly and professional tone as a rental agent.

    Question: {query}

    Context:
    {context}

    Answer:"""

    system_prompt = system_prompt.format(query=query, context=context)
    
    return system_prompt

def chatbot(query):
    """ Main function to interact with the chatbot. """
    
    prompt = generate_prompt(query)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

st.title("Rental Agent Chatbot")

user_query = st.text_input("Welcome to LivedIn Rentals. How can I help you today?")

if user_query:
    response = chatbot(user_query)
    st.write("Chatbot:", response)
    st.write("If you have any further questions, please feel free to ask!")


if __name__ == "__main__":
    pass
