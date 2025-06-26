import os
import io
import requests
from flask import Flask, request, render_template

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"


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
        "file": (file.filename, audio_stream, file.content_type)
    }

    data = {
        "model": "whisper-1",
        "language": "nl",
        "temperature": 0.0
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    try:
        # Whisper transcriptie
        response = requests.post(WHISPER_URL, headers=headers, files=files, data=data, timeout=90)
        response.raise_for_status()
        raw_transcript = response.json().get("text", "[Leeg resultaat]")

        # GPT-4o verbetering met medische instructie
        gpt_payload = {
            "model": "gpt-4o",
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": "Je bent een medisch taalmodel. Corrigeer en verbeter onderstaande transcriptie zodat deze grammaticaal correct is, coherent, en correct medisch Nederlands gebruikt."
                },
                {
                    "role": "user",
                    "content": raw_transcript
                }
            ]
        }

        gpt_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        gpt_response = requests.post(GPT_URL, headers=gpt_headers, json=gpt_payload, timeout=90)
        gpt_response.raise_for_status()
        transcript = gpt_response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        transcript = f"Fout tijdens transcriptie of verbetering: {str(e)}"

    return render_template("index.html", transcript=transcript)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
