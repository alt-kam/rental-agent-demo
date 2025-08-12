import os
import json

import numpy as np
import faiss
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from stripe_service import stripe_service
from datetime import datetime, timedelta, timezone

load_dotenv(override=True)
# Silence tokenizers parallelism warning in Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def load_data():
    """Loads reusable data."""
    
    index = faiss.read_index("rental_embeddings.index")

    with open("rental_chunks.json", "r", encoding='utf-8') as f:
        rental_chunks = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return index, rental_chunks, model

def get_relevant_answers(query, top_k: int = 3):
    """
    Chatbot function that retrieves relevant rental information based on the user's query.
    """

    index, rental_chunks, model = load_data()
    
    query_embedding = model.encode(query)

    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)

    return [
        rental_chunks[idx]
        for idx in indices[0]
        if idx != -1
    ][:top_k]

def format_history_for_prompt(history: list[str]) -> str:
    """Convert last few dialogue turns into a compact text block for the prompt."""
    if not history:
        return ""
    limited = history[-6:]
    lines = []
    for i, msg in enumerate(limited):
        prefix = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{prefix}: {msg}")
    return "\n".join(lines)


def generate_prompt(query, history_messages: list[dict] | None = None):
    """
    Generates a prompt for the chatbot based on the user's query and the relevant rental information.
    """
    
    relevant_chunks = get_relevant_answers(query)
    
    context = ""
    for chunk in relevant_chunks:
        context += f"Information: {chunk}\n\n"
    
    if not context.strip():
        context = "I'm sorry, but I couldn't find any relevant information to answer your question."

    conversation_history = [
        m.get("content", "") for m in (history_messages or [])
        if m.get("role") in {"user", "assistant"}
    ]

    system_prompt = f"""You are a helpful rental agent. 
    - Answer any question about the rental property based on the provided context.
    - Be concise, specific, and professional.
    - If missing info, say you don't have it. Avoid inventing details.
    
    Follow these steps:
    1. Welcome the user to LivedIn Rentals and ask them how you can help them today.
    2. If the user is asking about the property, use the context to answer the question.
    3. If the user wants to make a booking, ask the user for the dates they are interested in.
    4. When they give the dates, calculate the price of the stay for each night charge 400 SAR (Saudi Riyals).
    5. Ask the user if they would like to proceed to payment.
    6. If the user says yes, ask the user for email and phone number.
    7. IMPORTANT: When you have the email and phone number from the user, you MUST call the generate_checkout_link function with the guest_email, guest_phone, and nights parameters to create the Stripe checkout session.
    8. If the user says no to payment, ask if they have any other questions.
    9. If the user says no to questions, thank them for their time and say goodbye.

    TOOL USAGE:
    - Use the generate_checkout_link function when you have collected both email and phone number from the user
    - The function requires: guest_email (string), guest_phone (string), and nights (integer, default 3)

    Conversation History (most recent first):
    {format_history_for_prompt(conversation_history)}

    Current Question: {query}

    Context:
    {context}

    Answer:"""

    system_prompt = system_prompt.format(query=query, context=context)
    
    return system_prompt

def chatbot(query, history_messages: list[dict] | None = None):
    prompt = generate_prompt(query, history_messages)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    
    # Define the function schema for tool calling
    checkout_function = genai.protos.FunctionDeclaration(
        name="generate_checkout_link",
        description="Create a Stripe checkout session when user provides email and phone number for booking",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "guest_email": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Guest's email address"
                ),
                "guest_phone": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Guest's phone number"
                ),
                "nights": genai.protos.Schema(
                    type=genai.protos.Type.INTEGER,
                    description="Number of nights for the stay"
                )
            },
            required=["guest_email", "guest_phone"]
        )
    )
    
    tool = genai.protos.Tool(function_declarations=[checkout_function])
    model = genai.GenerativeModel('gemini-2.0-flash-exp', tools=[tool])

    try:
        response = model.generate_content(prompt)
        
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            if function_call.name == "generate_checkout_link":
                guest_email = function_call.args.get("guest_email", "")
                guest_phone = function_call.args.get("guest_phone", "")
                nights = function_call.args.get("nights", 3)

                result = generate_checkout_link(guest_email, guest_phone, nights)
                
                if result["success"]:
                    return f"{result['message']}\n\n[Complete your payment securely on Stripe]({result['payment_url']})"
                else:
                    return result["message"]

        return response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        return f"An error occurred while generating the response: {e}"


def generate_checkout_link(guest_email: str, guest_phone: str, nights: int = 3):
    """Create a Stripe Checkout session using booking details.
    
    Args:
        guest_email: Guest's email address
        guest_phone: Guest's phone number
        nights: Number of nights for the stay

    Returns:
        A dictionary containing the success status, message, and payment URL.
    """
    
    check_in_date = (datetime.now(timezone.utc) + timedelta(days=7)).date()
    check_out_date = check_in_date + timedelta(days=nights)
    amount_total = 400 * nights

    booking_details = {
        'property_name': 'LivedIn Rentals',
        'property_id': 'livedin_rentals',
        'guest_name': 'Guest',
        'guest_email': guest_email,
        'guest_phone': guest_phone,
        'check_in': check_in_date.isoformat(),
        'check_out': check_out_date.isoformat(),
        'original_price': 400,
        'amount': amount_total,
    }

    result = stripe_service.create_checkout_session(booking_details)
    if result.get('success'):
        url = result['session_url']
        return {
            "success": True,
            "message": f"Payment session created successfully. Total: {amount_total} SAR for {nights} nights.",
            "payment_url": url
        }
    else:
        return {
            "success": False,
            "message": f"Failed to create payment session: {result.get('error')}"
        }


#Streamlit UI
st.title("LivedIn Agent Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to LivedIn Rentals. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant reply using LLM (with potential function calling)
    assistant_reply = chatbot(user_input, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.write(assistant_reply)

    st.session_state["ready_for_payment"] = True


# Utility: new chat
with st.sidebar:
    if st.button("New chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to LivedIn Rentals. How can I help you today?"}
        ]


if __name__ == "__main__":
    pass
