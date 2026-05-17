import whisper
import tempfile
import os

# Load model once here so it stays in memory
model = whisper.load_model("base")

def handle_audio(audio_file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        audio_file.save(tf.name)
        result = model.transcribe(tf.name)
        os.remove(tf.name)
    return result.get("text", "")
