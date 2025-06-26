import os
import io
import datetime
import requests
from flask import Flask, request, render_template

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"

def call_gpt(messages, temperature=0.3):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "temperature": temperature,
        "messages": messages
    }
    response = requests.post(GPT_URL, headers=headers, json=data, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files.get("audio_file")
    verslag_type = request.form.get("verslag_type", "consult")
    if not file or file.filename == "":
        return render_template("index.html", transcript="⚠️ Geen bestand geselecteerd.")

    audio_stream = io.BytesIO(file.read())
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
        response = requests.post(WHISPER_URL, headers=headers, files=files, data=data, timeout=120)
        response.raise_for_status()
        raw_text = response.json().get("text", "")

        corrected = call_gpt([
            {"role": "system", "content": "Corrigeer deze transcriptie in correct medisch Nederlands."},
            {"role": "user", "content": raw_text}
        ])

        instruction = (
            f"Schrijf een gestructureerd medisch verslag voor een cardiologische {verslag_type}, "
            "met relevante secties zoals voorgeschiedenis, anamnese, klinisch onderzoek, aanvullend onderzoek, "
            "thuismedicatie, conclusie en beleid. Vermeld enkel wat expliciet gezegd wordt. Laat irrelevante of "
            "niet-gevulde onderdelen weg."
        )

        structured = call_gpt([
            {"role": "system", "content": instruction},
            {"role": "user", "content": corrected}
        ])

        advies = ""
        if verslag_type in ["raadpleging", "consult"]:
            advies = call_gpt([
                {
                    "role": "system",
                    "content": (
                        "Geef op basis van dit verslag concrete evidence-based aanbevelingen voor diagnose en behandeling, "
                        "inclusief Class of Recommendation en Level of Evidence volgens zowel ESC als AHA/ACC. "
                        "Voeg een link toe naar de richtlijn of PubMed. Beperk je tot relevante aanbevelingen. "
                        "Zet onderaan een kopje: 'Advies volgens ESC/AHA'."
                    )
                },
                {
                    "role": "user",
                    "content": structured
                }
            ])

        return render_template("index.html", transcript=structured, advies=advies)

    except Exception as e:
        return render_template("index.html", transcript=f"⚠️ Fout: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
