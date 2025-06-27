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

        template_instruction = ""
        today = datetime.date.today().strftime("%d-%m-%Y")

        if verslag_type == "TTE":
            template_instruction = f"""Gebruik uitsluitend onderstaand TTE-verslagformat. Vul alleen velden in die expliciet genoemd zijn. Vermijd incomplete zinnen. Indien waarden niet vermeld zijn, laat ze weg en herschrijf de zin grammaticaal correct. Beschrijf structuren als 'normaal' indien niet besproken. GEEN voorgeschiedenis, context of advies:
TTE ikv. (raadpleging/spoedconsult/consult) op {today}:
Linker ventrikel: (...)troof met EDD (...) mm, IVS (...) mm, PW (...) mm. Globale functie: (goed/licht gedaald/matig gedaald/ernstig gedaald) met LVEF (...)% (geschat/monoplane/biplane).
Regionaal: (geen kinetiekstoornissen/zone van hypokinesie/zone van akinesie)
Rechter ventrikel: (...)troof, globale functie: (...) met TAPSE (...) mm.
Diastole: (normaal/vertraagde relaxatie/dysfunctie graad 2/ dysfunctie graad 3) met E (...) cm/s, A (...) cm/s, E DT (...) ms, E' septaal (...) cm/s, E/E' (...). L-golf: (ja/neen).
Atria: LA (normaal/licht gedilateerd/sterk gedilateerd) (...) mm.
Aortadimensies: (normaal/gedilateerd) met sinus (...) mm, sinotubulair (...) mm, ascendens (...) mm.
Mitralisklep: morfologisch (normaal/sclerotisch/verdikt/prolaps/restrictief). insufficiëntie: (...), stenose: geen.
Aortaklep: (tricuspied/bicuspied), morfologisch (normaal/sclerotisch/mild verkalkt/matig verkalkt/ernstig verkalkt). Functioneel: insufficiëntie: geen, stenose: geen.
Pulmonalisklep: insufficiëntie: spoor, stenose: geen.
Tricuspiedklep: insufficiëntie: (...), geschatte RVSP: ( mmHg/niet opmeetbaar) + CVD (...) mmHg gezien vena cava inferior: (...) mm, variabiliteit: (...).
Pericard: (...)."""
        elif verslag_type == "TEE":
            template_instruction = f"""Gebruik uitsluitend onderstaand TEE-verslagformat. Vul alleen expliciet genoemde zaken in. Laat velden weg indien niet vermeld en herschrijf zinnen grammaticaal correct. Gebruik defaults enkel voor structuren die niet besproken zijn. GEEN voorgeschiedenis of advies:
Onderzoeksdatum: {today}
Bevindingen: TEE ONDERZOEK : 3D TEE met (Philips/GE) toestel
Indicatie: (...)
Afname mondeling consent: dr. Verbeke. Informed consent: patiënt kreeg uitleg over aard onderzoek, mogelijke resultaten en procedurele risico’s en verklaart zich hiermee akkoord.
Supervisie: dr (...)
Verpleegkundige: (...)
Anesthesist: dr. (...)
Locatie: endoscopie 3B
Sedatie met (Midazolam/Propofol) en topicale Xylocaine spray.
(Vlotte/moeizame) introductie TEE probe, (Vlot/moeizaam) verloop van onderzoek zonder complicatie.
VERSLAG:
- Linker ventrikel is (...), (niet/mild/matig/ernstig) gedilateerd en (...)contractiel (zonder/met) regionale wandbewegingstoornissen.
- Rechter ventrikel is (...), (...) gedilateerd en (...)contractiel.
- Atria zijn (...) gedilateerd.
- Linker hartoortje is (...), (geen/beperkt) spontaan contrast, zonder toegevoegde structuur. Snelheden: (...) cm/s.
- Interatriaal septum: (...)
- Mitralisklep: (...), insufficiëntie: (...), stenose: (...).
- Aortaklep: (...), insufficiëntie: (...), stenose: (...).
- Tricuspiedklep: (...), insufficiëntie: (...).
- Pulmonalisklep: (...).
- Aorta ascendens: (...).
- Pulmonale arterie: (...).
- VCI/levervenes: (...).
- Pericard: (...)."""
        else:
            template_instruction = (
                f"Schrijf een gestructureerd medisch verslag voor een cardiologische {verslag_type}, "
                "met relevante secties zoals voorgeschiedenis, anamnese, klinisch onderzoek, aanvullend onderzoek, "
                "thuismedicatie, conclusie en beleid. Vermeld enkel wat expliciet gezegd wordt. Laat irrelevante of "
                "niet-gevulde onderdelen weg."
            )

        structured = call_gpt([
            {"role": "system", "content": template_instruction},
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

        if verslag_type in ["TTE", "TEE"]:
            return render_template("index.html", transcript=structured)

        return render_template("index.html", transcript=structured, advies=advies)

    except Exception as e:
        return render_template("index.html", transcript=f"⚠️ Fout: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
