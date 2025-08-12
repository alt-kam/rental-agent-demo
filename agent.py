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
import re

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
        return "I'm sorry, but I couldn't find any relevant information to answer your question."

    conversation_history = [
        m.get("content", "") for m in (history_messages or [])
        if m.get("role") in {"user", "assistant"}
    ]

    system_prompt = f"""You are a helpful rental agent. Answer only using the provided Context.
    - Be concise, specific, and professional.
    - If missing info, say you don't have it.
    - Avoid inventing details.

    Conversation History (most recent first):
    {format_history_for_prompt(conversation_history)}

    Current Question: {query}

    Context:
    {context}

    Answer:"""

    system_prompt = system_prompt.format(query=query, context=context)
    
    return system_prompt

def chatbot(query, history_messages: list[dict] | None = None):
    """ Main function to interact with the chatbot. """
    
    prompt = generate_prompt(query, history_messages)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

def extract_property_name(answer_text: str) -> str:
    """Best-effort extraction of a property name from the chatbot answer."""
    if not answer_text:
        return "Rental Booking"
    # Try quoted name first
    if '"' in answer_text:
        parts = answer_text.split('"')
        if len(parts) >= 3 and len(parts[1].strip()) >= 3:
            return parts[1].strip()
    # Fallback: take words before ' is '
    lower = answer_text.lower()
    if " is " in lower:
        idx = lower.index(" is ")
        candidate = answer_text[:idx].strip()
        if len(candidate) >= 3:
            return candidate[:80]
    # Default title-case of first few words
    return answer_text.split(".")[0][:80]


def proceed_to_payment_from_answer(answer_text: str):
    """Create a Stripe Checkout session using inferred booking details and display the link."""
    st.markdown("---")
    st.subheader("Redirecting to secure payment")
    st.caption("Test mode - Stripe sandbox")

    property_name = extract_property_name(answer_text)

    # Simple defaults for demo: 3 nights next week at $450/night
    check_in_date = (datetime.now(timezone.utc) + timedelta(days=7)).date()
    check_out_date = check_in_date + timedelta(days=3)
    amount_total = 450 * 3

    booking_details = {
        'property_name': property_name,
        'property_id': property_name.lower().replace(' ', '_')[:40],
        'guest_name': 'Guest',
        'guest_email': None,
        'check_in': check_in_date.isoformat(),
        'check_out': check_out_date.isoformat(),
        'original_price': 500,
        'amount': amount_total,
        'property_images': []
    }

    result = stripe_service.create_checkout_session(booking_details)
    if result.get('success'):
        url = result['session_url']
        st.success("Payment session created.")
        st.markdown(f"[Complete your payment securely on Stripe]({url})")
    else:
        st.error(f"Failed to create payment session: {result.get('error')}")


# -------- Booking helpers (pricing + parsing) --------

def detect_payment_intent(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    keywords = [
        "proceed to payment",
        "proceed to pay",
        "pay now",
        "pay",
        "payment",
        "checkout",
        "check out",
        "book now",
        "book it",
        "book this",
        "please book",
        "go ahead and book",
        "reserve",
        "confirm booking",
        "confirm my booking",
        "confirm my bookings",
        "confirm reservation",
        "confirm my reservation",
        "proceed with booking",
        "confirm and pay",
    ]
    return any(k in text_lower for k in keywords)


MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


def _infer_year(month: int, day: int) -> int:
    today = datetime.now(timezone.utc).date()
    year = today.year
    try_date = datetime(year, month, day).date()
    if try_date < today:
        return year + 1
    return year


def parse_date_fragment(fragment: str) -> datetime | None:
    frag = fragment.strip().lower()
    # dd/mm/yy or dd/mm/yyyy
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", frag)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day)
        except Exception:
            return None
    # 10th august, 10 aug
    m = re.match(r"(\d{1,2})(st|nd|rd|th)?\s+([a-zA-Z]+)", frag)
    if m:
        day = int(m.group(1))
        mon_name = m.group(3).lower()
        if mon_name in MONTHS:
            month = MONTHS[mon_name]
            year = _infer_year(month, day)
            try:
                return datetime(year, month, day)
            except Exception:
                return None
    # august 10
    m = re.match(r"([a-zA-Z]+)\s+(\d{1,2})", frag)
    if m:
        mon_name = m.group(1).lower()
        if mon_name in MONTHS:
            day = int(m.group(2))
            month = MONTHS[mon_name]
            year = _infer_year(month, day)
            try:
                return datetime(year, month, day)
            except Exception:
                return None
    return None


def parse_dates_and_nights(text: str):
    """Extract a date range and/or nights from free text.
    Returns (check_in: date|None, check_out: date|None, nights: int|None).
    """
    if not text:
        return None, None, None
    t = text.lower()

    # nights like '6 nights'
    nights = None
    m = re.search(r"(\d{1,2})\s*nights?", t)
    if m:
        nights = int(m.group(1))

    # range 'from X to Y' or 'X to Y'
    m = re.search(r"from\s+([^\n]+?)\s+to\s+([^\n]+)", t)
    if not m:
        m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}(st|nd|rd|th)?\s+[a-zA-Z]+|[a-zA-Z]+\s+\d{1,2})\s+to\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}(st|nd|rd|th)?\s+[a-zA-Z]+|[a-zA-Z]+\s+\d{1,2})",
                         t)
        if m:
            start_raw, end_raw = m.group(1), m.group(3)
            start_dt = parse_date_fragment(start_raw)
            end_dt = parse_date_fragment(end_raw)
        else:
            start_dt = end_dt = None
    else:
        start_dt = parse_date_fragment(m.group(1))
        end_dt = parse_date_fragment(m.group(2))

    # 'starting from X' with nights
    if not (start_dt and end_dt):
        m = re.search(r"starting\s+from\s+([^\n]+)", t)
        if m:
            start_dt = parse_date_fragment(m.group(1))
            if start_dt and nights:
                end_dt = start_dt + timedelta(days=nights)

    # single date with nights
    if not (start_dt and end_dt):
        # try any single date mention
        for pattern in [r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}", r"\d{1,2}(st|nd|rd|th)?\s+[a-zA-Z]+", r"[a-zA-Z]+\s+\d{1,2}"]:
            m = re.search(pattern, t)
            if m:
                start_dt = parse_date_fragment(m.group(0))
                break
        if start_dt and nights:
            end_dt = start_dt + timedelta(days=nights)

    # finalize nights
    if start_dt and end_dt:
        nights = (end_dt.date() - start_dt.date()).days

    return (start_dt.date() if start_dt else None,
            end_dt.date() if end_dt else None,
            nights)


def nightly_rate_for_property(name: str | None) -> int:
    n = (name or "").lower()
    if "oasis" in n:
        return 450
    if "luxury" in n:
        return 550
    return 350


def compute_price(property_name: str | None, nights: int | None) -> dict | None:
    if not nights or nights <= 0:
        return None
    rate = nightly_rate_for_property(property_name)
    subtotal = rate * nights
    cleaning_fee = 60
    service_fee = round(subtotal * 0.08)
    total = subtotal + cleaning_fee + service_fee
    return {
        'rate': rate,
        'nights': nights,
        'subtotal': subtotal,
        'cleaning_fee': cleaning_fee,
        'service_fee': service_fee,
        'total': total,
    }


st.title("Rental Agent Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to LivedIn Rentals. How can I help you today?"}
    ]

# Display chat history with lightweight pricing inline when possible
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If assistant message and prior user mentioned dates/nights, show quick price
        if msg["role"] == "assistant":
            # Look back to find last user message before this assistant reply
            idx = st.session_state.messages.index(msg)
            user_before = None
            for j in range(idx - 1, -1, -1):
                if st.session_state.messages[j]["role"] == "user":
                    user_before = st.session_state.messages[j]["content"]
                    break
            if user_before:
                # extract property name from assistant text, parse dates from user text
                property_name = extract_property_name(msg["content"])
                ci, co, nights = parse_dates_and_nights(user_before)
                price = compute_price(property_name, nights)
                if price:
                    st.caption(
                        f"Estimated total for {nights} nights at ${price['rate']}/night + fees: "
                        f"${price['total']}. Say 'proceed to payment' to pay."
                    )

# Chat input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 1) If user wants to pay, skip LLM and go straight to checkout creation
    if detect_payment_intent(user_input):
        last_assistant_text = next((m["content"] for m in reversed(st.session_state.messages)
                                    if m["role"] == "assistant"), "")
        with st.chat_message("assistant"):
            st.write("Great, creating your secure checkout now...")
            proceed_to_payment_from_answer(last_assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": "I have created a secure Stripe checkout link for you above."})
    else:
        # 2) Otherwise, generate a normal assistant reply
        assistant_reply = chatbot(user_input, st.session_state.messages)

        # If dates/nights detected, compose a pricing answer instead of a generic LLM response
        ci, co, nights = parse_dates_and_nights(user_input)
        if nights:
            # Prefer the most recent assistant property suggestion before this turn
            last_assistant_text = next((m["content"] for m in reversed(st.session_state.messages)
                                        if m["role"] == "assistant"), "")
            property_name = extract_property_name(last_assistant_text or assistant_reply)
            price = compute_price(property_name, nights)
            if price:
                price_text = (
                    f"For {property_name} from {ci} to {co} ({nights} nights):\n"
                    f"- ${price['rate']}/night\n"
                    f"- Cleaning fee: ${price['cleaning_fee']}\n"
                    f"- Service fee: ${price['service_fee']}\n"
                    f"Estimated total: ${price['total']}.\n\n"
                    f"Would you like me to proceed to payment?"
                )
                st.session_state.messages.append({"role": "assistant", "content": price_text})
                with st.chat_message("assistant"):
                    st.write(price_text)
            else:
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                with st.chat_message("assistant"):
                    st.write(assistant_reply)
        else:
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
