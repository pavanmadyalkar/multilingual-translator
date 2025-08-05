import streamlit as st
# import evaluate
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
# import json
import torch

# ------------------------
# Load reference dataset
# ------------------------
# with open("./reference.json", "r", encoding="utf-8") as f:
#     reference_data = json.load(f)
#     print(reference_data)

# ------------------------
# Language mapping
# ------------------------
lang_map = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Russian": "ru"
}
langs = list(lang_map.keys())

# ------------------------
# Caching Model Loading
# ------------------------
@st.cache_resource
def load_translation_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# BLEU scorer
# @st.cache_resource
# def load_bleu():
#     return evaluate.load("sacrebleu")

# ------------------------
# Streamlit UI
# ------------------------
st.title("🌍 Multilingual Translator with Speech")

# Dropdowns
col1, col2 = st.columns(2)
with col1:
    sourceLang = st.selectbox("From", langs, label_visibility="collapsed")
    input_text = st.text_area("", placeholder="Enter text here...", height=80)
with col2:
    targetLang = st.selectbox("To", [lang for lang in langs if lang != sourceLang], label_visibility="collapsed")
    output_text = st.empty()

# Build model name
model_name = f"Helsinki-NLP/opus-mt-{lang_map[sourceLang]}-{lang_map[targetLang]}"

# Load cached model
tokenizer, model = load_translation_model(model_name)

if st.button("Translate"):
    if input_text.strip():
        # Translation
        tokens = tokenizer(input_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            translation = model.generate(**tokens)
        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

        # Show result
        st.subheader(f"Text to speech in ({targetLang}):")
        st.write(translated_text)
        
        # Update in col2
        output_text.text_area("Translated text", value=translated_text, disabled=True, height=80)

        # Text-to-Speech
        tts = gTTS(translated_text, lang=lang_map.get(targetLang, "en"))
        tts.save("output.mp3")
        st.audio("output.mp3")

        # BLEU evaluation
        # bleu = load_bleu()
        # reference = reference_data.get(lang_map[sourceLang], {}).get(input_text, {}).get(lang_map[targetLang], None)

        # if reference:
        #     results = bleu.compute(predictions=[translated_text], references=[[reference]])
        #     st.subheader("📊 BLEU Score")
        #     st.write(f"{results['score']:.2f}")
        # else:
        #     st.info("ℹ️ No reference available for BLEU evaluation.")
