import streamlit as st
import numpy as np
import faiss
import tempfile
import re
import os
import pickle

from gtts import gTTS
from groq import Groq
from sentence_transformers import SentenceTransformer
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
from dotenv import load_dotenv

load_dotenv()

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SPECTRA Physics Tutor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background-color: #0f1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_voice" not in st.session_state:
    st.session_state.last_voice = None

# -----------------------------
# LOAD NOTES
# -----------------------------
with open("clean_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"\s+", " ", text)

# -----------------------------
# IMPROVED CHUNKING
# -----------------------------
def chunk_text(text, size=800, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        start += size - overlap

    return chunks

chunks = chunk_text(text)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# EMBEDDINGS (FIXED - NORMALIZED)
# -----------------------------
@st.cache_resource
def create_embeddings(chunks):
    emb = model.encode(chunks)
    emb = np.array(emb).astype("float32")

    # IMPORTANT FIX
    faiss.normalize_L2(emb)

    return emb

embeddings = create_embeddings(chunks)

# -----------------------------
# FAISS INDEX (COSINE SIMILARITY FIX)
# -----------------------------
@st.cache_resource
def build_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    return index

index = build_index(embeddings)

# -----------------------------
# RETRIEVAL (FIXED - NO REPETITION)
# -----------------------------
def retrieve_context(question, k=3):

    q_emb = model.encode([question])
    q_emb = np.array(q_emb).astype("float32")

    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, k * 5)

    results = []
    seen = set()

    for i, idx in enumerate(indices[0]):

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        # avoid duplicates
        if chunk in seen:
            continue

        seen.add(chunk)
        results.append(chunk)

        if len(results) == k:
            break

    return "\n".join(results)

# -----------------------------
# GROQ API
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# LAYOUT
# -----------------------------
left, center, right = st.columns([1, 3, 1])

# -----------------------------
# LEFT (AVATAR)
# -----------------------------
with left:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=120)
    st.markdown("### 🧠 SPECTRA AI")
    st.caption("Physics Tutor")

# -----------------------------
# RIGHT (TOOLS)
# -----------------------------
with right:
    st.markdown("### ⚙️ Tools")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# -----------------------------
# CENTER (MAIN CHAT)
# -----------------------------
with center:

    st.title("🧠 SPECTRA Physics Tutor")
    st.write("Ask questions using text or voice")

    # -------------------------
    # VOICE INPUT
    # -------------------------
    st.subheader("🎤 Voice Input")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="recorder"
    )

    voice_text = None

    if audio:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(audio["bytes"])
                temp_audio_path = temp_audio.name

            sound = AudioSegment.from_file(temp_audio_path, format="webm")
            wav_path = temp_audio_path.replace(".webm", ".wav")
            sound.export(wav_path, format="wav")

            recognizer = sr.Recognizer()

            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio_data)

                st.success(f"You said: {voice_text}")
                st.session_state.last_voice = voice_text

        except Exception as e:
            st.error(f"Audio error: {e}")

    # -------------------------
    # TEXT INPUT
    # -------------------------
    user_input = st.chat_input("💬 Ask your physics question...")

    if voice_text is not None:
        user_input = voice_text

    # -------------------------
    # CHAT LOGIC
    # -------------------------
    if user_input:

        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        context = retrieve_context(user_input)

        prompt = f"""
You are a strict Physics tutor AI.

RULES:
- Use ONLY given context
- If context is irrelevant, say you don't know
- Do NOT repeat previous answers
- Keep answer max 5 lines

CONTEXT:
{context}

QUESTION:
{user_input}
"""

        with st.spinner("🧠 Thinking..."):

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = completion.choices[0].message.content

        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

        # -------------------------
        # TEXT TO SPEECH
        # -------------------------
        try:
            tts = gTTS(answer)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                audio_path = fp.name

            tts.save(audio_path)

            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"Voice output error: {e}")

    # -----------------------------
    # CHAT HISTORY
    # -----------------------------
    st.markdown("---")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)