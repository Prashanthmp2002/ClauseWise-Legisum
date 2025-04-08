import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
import base64
import os

# Function to set background image
def add_bg_from_local(image_file):
    if os.path.exists(image_file):  # Ensure the file exists
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image not found! Using default background.")

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Function to summarize text
def summarize(text):
    try:
        input_text = "summarize: " + text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Function for text-to-speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("summary.mp3")

# Set background image
add_bg_from_local("img.avif")

# Custom CSS styling
st.markdown("""
    <style>
    label {
        color: #FFD700 !important;  /* Gold color */
        font-weight: bold !important;
        font-size: 18px !important;
    }
    textarea {
        background-color: #f0f0f5 !important;  /* Light grey background */
        border: 2px solid #8A2BE2 !important;  /* Purple border */
        border-radius: 10px !important;
        color: #000000 !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FFFFFF;">ClauseWise Legisum</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Text input
text_input = st.text_area("Paste your Terms and Conditions here:", height=300)

# Generate Summary button
generate_btn = st.button("✨ Generate Summary")

# Summary Generation Logic
if generate_btn:
    if text_input.strip():
        summary = summarize(text_input)

        # Stylized summary heading and box
        st.markdown(
            "<h2 style='color:#FFD700;'>Summary:</h2>",  # Gold-colored heading
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style='
                background-color: #ffffffcc;
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #8A2BE2;
                color: #000000;
                font-size: 16px;
                font-weight: 500;
            '>{summary}</div>
            """,
            unsafe_allow_html=True
        )

        # Convert to speech
        text_to_speech(summary)

        # Audio player
        with open("summary.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Please paste the Terms and Conditions text first.")
