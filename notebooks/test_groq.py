# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# [span_0](start_span)Charger la cle depuis .env[span_0](end_span)
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

# [span_1](start_span)Creer le client Groq[span_1](end_span)
client = Groq(api_key=api_key)

# [span_2](start_span)Premier appel : question simple[span_2](end_span)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. "
                    "Reponds en francais simple. "
                    "Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

# [span_3](start_span)Afficher la reponse[span_3](end_span)
print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

# --- Étape 3.3 : Test avec le format SénSanté ---

print("\n=== Test avec format SénSanté ===")

# On définit un prompt "système" plus précis pour le contexte médical
system_prompt_medical = """Tu es un assistant médical sénégalais. 
Tu reçois un diagnostic et des données patient. 
Explique le résultat en français simple, comme un médecin parlerait à son patient. 
Sois rassurant mais recommande une consultation. 
Maximum 3 phrases. 
Ne fais JAMAIS de diagnostic toi-même."""

# On simule les données qu'Awa (l'agente de santé) pourrait saisir
user_data = """Patient Femme, 28 ans, région Dakar
Symptômes : température 39.5, toux, fatigue, maux de tête
Diagnostic du modèle : paludisme (probabilité 72%)

Explique ce résultat au patient."""

response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": system_prompt_medical},
        {"role": "user", "content": user_data}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Explication SénSanté ===")
print(response2.choices[0].message.content)