
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

        today = datetime.date.today().strftime("%d-%m-%Y")
        if verslag_type == "TTE":
            template_instruction = f"""Gebruik het volgende TTE-sjabloon en vul aan met enkel wat vermeld wordt. Laat incomplete zinnen weg. Gebruik 'normaal' voor niet-vermelde structuren:
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
            template_instruction = f"""Gebruik onderstaand TEE sjabloon en vul enkel aan met relevante info. Laat onvolledige zinnen weg. Vul defaults in waar nodig:
Onderzoeksdatum: {today}
Bevindingen: TEE ONDERZOEK : 3D TEE met (Philips/GE) toestel
Indicatie: (klepfunctie nl: .../vermoeden endocarditis/cardioversie).
Afname mondeling consent: dr. Verbeke. Informed consent: patiënt kreeg uitleg over aard onderzoek, mogelijke resultaten en procedurele risico’s en verklaart zich hiermee akkoord.
Supervisie: dr (Dujardin/Bergez/Anné/de Ceuninck/Vanhaverbeke/Gillis/Van de Walle/Muyldermans)
Verpleegkundige: (Sieglien/Nathalie/Tom/Bruno)
Anesthesist: dr. (naam)
Locatie: endoscopie 3B
Sedatie met (Midazolam/Propofol) en topicale Xylocaine spray.
(Vlotte/moeizame) introductie TEE probe, (Vlot/moeizaam) verloop van onderzoek zonder complicatie.
VERSLAG:
- Linker ventrikel is (eutroof/hypertroof), (niet/mild/matig/ernstig) gedilateerd en (normocontractiel/licht hypocontractiel/matig hypocontractiel/ernstig hypocontractiel) (zonder/met) regionale wandbewegingstoornissen.
- Rechter ventrikel is (eutroof/hypertroof), (niet/mild/matig/ernstig) gedilateerd en (normocontractiel/licht hypocontractiel/matig hypocontractiel/ernstig hypocontractiel).
- De atria zijn (niet/licht/matig/sterk) gedilateerd.
..."""

        else:
            template_instruction = f"Schrijf een beknopt medisch verslag in het Nederlands."

        structured = call_gpt([
            {"role": "system", "content": template_instruction},
            {"role": "user", "content": corrected}
        ])

        return render_template("index.html", transcript=structured, raw=raw_text)

    except Exception as e:
        return render_template("index.html", transcript=f"⚠️ Fout: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
