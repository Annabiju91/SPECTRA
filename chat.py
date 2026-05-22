import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------- CONFIG ----------------
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


st.set_page_config(page_title="Physics AI Tutor", page_icon="📘")

st.title("📘 Physics AI Tutor (Momentum & More)")

# ---------------- SCRAPER ----------------
@st.cache_data
def load_data():
    url = "https://www.physicsclassroom.com/class/momentum"
    r = requests.get(url)

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text().split())

    words = text.split()
    chunks = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return chunks, model, index


chunks, model, index = load_data()

# ---------------- QA FUNCTION ----------------
def ask(question):
    q_emb = model.encode([question])
    _, I = index.search(np.array(q_emb), k=3)

    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
You are a strict physics teacher.

Give a clear, exam-ready answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content


# ---------------- CHAT UI ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask a physics question...")

if user_input:
    answer = ask(user_input)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("AI", answer))

for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"🧑‍🎓 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")