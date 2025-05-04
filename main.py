# main.py
import streamlit as st
from groq_translation import Translator
from voice_cloning import VoiceCloner
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import numpy as np

# Create a directory for voice profiles if it doesn't exist
VOICE_PROFILE_DIR = "voice_profiles"
os.makedirs(VOICE_PROFILE_DIR, exist_ok=True)

# Initialize models
translator = Translator()
voice_cloner = VoiceCloner()
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Set up Streamlit app
st.set_page_config(page_title="Fast Translate", page_icon="🎙️")
st.title("Fast Translate")
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for voice profile
with st.sidebar:
    st.header("Voice Profile")
    
    # Check if a voice profile already exists
    voice_profile_path = os.path.join(VOICE_PROFILE_DIR, "user_voice.mp3")
    if os.path.exists(voice_profile_path):
        voice_cloner.add_reference_voice(voice_profile_path)
        st.success("Using existing voice profile!")
        if st.button("Upload new voice profile"):
            voice_sample = st.file_uploader("Upload voice sample (MP3)", type=["mp3"])
            if voice_sample:
                with open(voice_profile_path, "wb") as f:
                    f.write(voice_sample.getvalue())
                voice_cloner.add_reference_voice(voice_profile_path)
                st.success("Voice profile updated successfully!")
    else:
        st.info("Upload a voice sample to clone your voice")
        voice_sample = st.file_uploader("Upload voice sample (MP3)", type=["mp3"])
        if voice_sample:
            with open(voice_profile_path, "wb") as f:
                f.write(voice_sample.getvalue())
            voice_cloner.add_reference_voice(voice_profile_path)
            st.success("Voice profile saved successfully!")

# Main content
st.header("Translate Your Speech")
st.info("Record your speech in English, and it will be translated to Japanese with your voice.")

# Language selection
target_language = st.selectbox(
    "Target Language",
    ["Japanese", "Spanish", "French", "German", "Korean"],
    index=0
)

# Audio recording
audio_bytes = audio_recorder(
    text="Record your speech",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x",
)

if audio_bytes:
    with st.spinner("Processing..."):
        # Save recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name

        # Transcribe the audio using faster-whisper
        segments, _ = whisper_model.transcribe(temp_audio_path)
        transcribed_text = " ".join([segment.text for segment in segments])
        
        # Translate the text
        translated_text = translator.translate(transcribed_text, target_language)
        
        # Generate speech with cloned voice
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_output:
            # Pass the language to the voice cloner
            voice_cloner.apply_voice_style(translated_text, temp_output.name, language=target_language.lower())
            
            # Display results
            st.subheader("Original Text")
            st.write(transcribed_text)
            
            st.subheader("Translation")
            st.write(translated_text)
            
            st.subheader("Translated Speech")
            st.audio(temp_output.name, format="audio/mp3")
            
            # Clean up temporary files
            os.unlink(temp_audio_path)
            os.unlink(temp_output.name)