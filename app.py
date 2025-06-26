import os
import io
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
    payload = {
        "model": "gpt-4o",
        "temperature": temperature,
        "messages": messages
    }
    response = requests.post(GPT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files.get("audio_file")
    type_verslag = request.form.get("verslag_type", "consult")
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
        # STAP 1: Transcriptie (Whisper)
        response = requests.post(WHISPER_URL, headers=headers, files=files, data=data, timeout=90)
        response.raise_for_status()
        raw_text = response.json().get("text", "")

        # STAP 2: Terminologiecorrectie
        corrected = call_gpt([
            {"role": "system", "content": "Corrigeer deze transcriptie voor correct medisch Nederlands, grammatica, en terminologie."},
            {"role": "user", "content": raw_text}
        ])

        # STAP 3: Validatie
        validated = call_gpt([
            {"role": "system", "content": "Evalueer de tekst op inconsistenties (geslacht, eenheden, medische plausibiliteit), en verbeter waar nodig."},
            {"role": "user", "content": corrected}
        ])

        # STAP 4: Gestructureerd verslag
        template_prompt = f"Gestructureerd verslag ({type_verslag}). Splits op in Bevindingen, Conclusie en Aanbeveling. Gebruik medisch Nederlands."
        structured = call_gpt([
            {"role": "system", "content": template_prompt},
            {"role": "user", "content": validated}
        ])

        # STAP 5: Advies (ICD-10 + richtlijnen)
        advies = call_gpt([
            {"role": "system", "content": "Stel een diagnose op basis van dit verslag, geef een ICD-10 code, aanbeveling volgens richtlijnen (met Class en Level of Evidence) en een link naar de bron."},
            {"role": "user", "content": structured}
        ])

        return render_template("index.html", transcript=structured, advies=advies, raw=raw_text)

    except Exception as e:
        return render_template("index.html", transcript=f"⚠️ Fout: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
