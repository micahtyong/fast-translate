from gtts import gTTS
import os
import tempfile

class VoiceCloner:
    def __init__(self):
        self.reference_audio_path = None

    def add_reference_voice(self, audio_path):
        """Add a reference voice sample (not used with gTTS)"""
        pass

    def synthesize_speech(self, text, output_path, language="en"):
        """Generate speech from text using gTTS with male voice"""
        # For Japanese, use a male voice
        if language.lower() == "japanese":
            tts = gTTS(text=text, lang='ja', slow=False)
        else:
            tts = gTTS(text=text, lang=language, slow=False)
        # Save to file
        tts.save(output_path)

    def apply_voice_style(self, text, output_path, language="en"):
        """Apply the voice style to the translated text"""
        self.synthesize_speech(text, output_path, language) 