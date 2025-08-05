# 🌍 Multilingual Translator with Speech

A simple **Streamlit web app** that performs **multilingual machine translation** using Hugging Face's MarianMT models and provides **speech output** using Google Text-to-Speech (gTTS).

---

## 🚀 Features
- Translate text between multiple languages:
  - English
  - French
  - German
  - Russian
- Dynamic language selection (`From` and `To`).
- Text-to-Speech (listen to the translated text).
- Built with **Streamlit** for an interactive UI.
- Uses **Helsinki-NLP MarianMT pretrained models** for translation.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – Web app framework  
- [Transformers](https://huggingface.co/transformers/) – Hugging Face MarianMT models  
- [Torch](https://pytorch.org/) – Deep learning backend  
- [gTTS](https://pypi.org/project/gTTS/) – Speech synthesis  
- [SentencePiece](https://github.com/google/sentencepiece) – Tokenizer used by MarianMT  

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/multilingual-translator.git
   cd multilingual-translator

2. **Create a virtual environment (recommended)**  
   ```bash
   conda create -n streamlit_env python=3.10
   conda activate streamlit_env

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

4. **Install dependencies**  
   ```bash
   streamlit run app.py

---

##  Future Improvements
- Add BLEU or semantic similarity metrics for evaluation.
- Support additional languages.
- Add sentiments to audio output like, Happy, sad, whisper etc

