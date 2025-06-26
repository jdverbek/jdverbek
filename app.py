import os
import io
import tempfile
import requests
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio_file" not in request.files:
        return "No file part", 400

    file = request.files["audio_file"]
    if file.filename == "":
        return "No selected file", 400

    audio_bytes = file.read()
    audio_stream = io.BytesIO(audio_bytes)

    files = {
        "file": (file.filename, audio_stream, file.content_type),
        "model": (None, "whisper-1"),
        "language": (None, "nl"),
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, files=files, timeout=60)
        response.raise_for_status()
        transcript = response.json().get("text", "[Leeg resultaat]")
    except Exception as e:
        transcript = f"Fout tijdens transcriptie: {str(e)}"

    return render_template("index.html", transcript=transcript)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

