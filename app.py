import streamlit as st
import numpy as np
import faiss
import tempfile
import re
import os

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
AudioSegment.ffprobe   = which("ffprobe")# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Physics Tutor",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🧠 AI Physics Tutor")
st.write("Ask questions using text or voice")

# -----------------------------
# CHAT HISTORY
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# LOAD CLEAN NOTES
# -----------------------------

with open("clean_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# CLEAN TEXT
# -----------------------------

text = re.sub(r"\s+", " ", text)

# -----------------------------
# CHUNKING
# -----------------------------

def chunk_text(text, size=500):

    chunks = []

    for i in range(0, len(text), size):

        chunk = text[i:i + size]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    return chunks

chunks = chunk_text(text)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

@st.cache_resource
def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    return embeddings

embeddings = create_embeddings(chunks)

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------

@st.cache_resource
def create_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

index = create_index(embeddings)

# -----------------------------
# RETRIEVE CONTEXT
# -----------------------------

def retrieve_context(question, k=3):

    q_embedding = model.encode([question])

    q_embedding = np.array(q_embedding).astype("float32")

    distances, indices = index.search(q_embedding, k)

    retrieved_chunks = []

    for i in indices[0]:

        chunk = chunks[i]

        if len(chunk.strip()) > 40:
            retrieved_chunks.append(chunk)

    return "\n".join(retrieved_chunks)

# -----------------------------
# GROQ CLIENT
# -----------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# VOICE INPUT
# -----------------------------

# -----------------------------
# VOICE INPUT
# -----------------------------

st.subheader("🎤 Voice Input")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key="recorder"
)

voice_text = ""

if audio:

    try:

        # SAVE AUDIO AS WEBM
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:

            temp_audio.write(audio["bytes"])

            temp_audio_path = temp_audio.name

        # CONVERT WEBM TO WAV
        sound = AudioSegment.from_file(
            temp_audio_path,
            format="webm"
        )

        wav_path = temp_audio_path.replace(".webm", ".wav")

        sound.export(
            wav_path,
            format="wav"
        )

        # SPEECH RECOGNITION
        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:

            audio_data = recognizer.record(source)

            voice_text = recognizer.recognize_google(audio_data)

            st.success(f"You said: {voice_text}")

    except Exception as e:

        st.error(f"Audio error: {e}")

# -----------------------------
# TEXT INPUT
# -----------------------------

user_input = st.chat_input("Ask your physics question...")

# -----------------------------
# PRIORITIZE VOICE INPUT
# -----------------------------

if voice_text:
    user_input = voice_text

# -----------------------------
# MAIN CHAT LOGIC
# -----------------------------

if user_input:

    # USER MESSAGE
    st.chat_message("user").markdown(user_input)

    st.session_state.chat_history.append(
        ("user", user_input)
    )

    # RETRIEVE CONTEXT
    context = retrieve_context(user_input)

    # PROMPT
    prompt = f"""
You are a STRICT Physics tutor AI.

RULES:
- Answer ONLY from the provided context.
- DO NOT use outside knowledge.
- DO NOT generate links.
- DO NOT mention websites.
- Keep answers short and simple.
- Maximum 5 lines.

If answer is not found, say:
"This topic is not available in the notes."

CONTEXT:
{context}

QUESTION:
{user_input}
"""

    # AI RESPONSE
    with st.spinner("Thinking..."):

        completion = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = completion.choices[0].message.content

    # SHOW RESPONSE
    st.chat_message("assistant").markdown(answer)

    st.session_state.chat_history.append(
        ("assistant", answer)
    )

    # -----------------------------
    # TEXT TO SPEECH
    # -----------------------------

    try:

        tts = gTTS(answer)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:

            audio_path = fp.name

        tts.save(audio_path)

        audio_file = open(audio_path, "rb")

        audio_bytes = audio_file.read()

        st.audio(
            audio_bytes,
            format="audio/mp3",
            autoplay=True
        )

    except Exception as e:

        st.error(f"Voice output error: {e}")

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

st.divider()

st.subheader("💬 Chat History")

for role, message in st.session_state.chat_history:

    with st.chat_message(role):

        st.markdown(message)