# 🌐 Telugu to English Neural Machine Translator 🇮🇳➡️🇬🇧
> A Sequence-to-Sequence (Seq2Seq) LSTM-based Neural Machine Translation (NMT) model for translating Telugu sentences into English using TensorFlow and Keras.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)


---

## 🧠 Problem Statement

Telugu is one of the major languages spoken in India but lacks robust translation tools when compared to high-resource languages like English. The goal of this project is to create a basic Neural Machine Translation (NMT) system that can convert Telugu sentences into English using a deep learning model. This serves as a foundational prototype for building a larger, production-grade translation tool.

---

## ✨ Features

- ✅ End-to-End Telugu ➡️ English translation system
- 📚 Custom tokenizer with <start> and <end> token handling
- 🔁 LSTM-based Seq2Seq architecture
- 🧪 Inference with greedy decoding
- 🛠️ Easily customizable and extensible
- ⚙️ Trained on small manually-curated dataset (demo-friendly)

---

## 🛠️ Technologies Used

| Technology       | Description                          |
|------------------|--------------------------------------|
| Python           | Programming language                 |
| TensorFlow/Keras | Deep learning framework              |
| NumPy            | Numerical operations                 |
| Jupyter Notebook | Development interface (optional)     |
| Matplotlib       | (Optional) For plotting loss curves  |

---

## 📚 Dataset

Sample pairs:

| Telugu                             | English                      |
|------------------------------------|-------------------------------|
| నేను ఇంగ్లీష్ మాట్లాడగలుగుతాను     | I can speak English           |
| ఈ రోజు చాలా మంచిది                 | Today is very good            |
| మీరు ఎలా ఉన్నారు?                 | How are you?                  |

---

## 🔍 Procedure

1. Add `<start>` and `<end>` tokens to target (English) sentences
2. Tokenize both Telugu (source) and English (target) sentences
3. Pad sequences to a uniform length
4. Build a Seq2Seq model with LSTM encoder and decoder
5. Train the model using categorical crossentropy
6. Use encoder-decoder inference to translate new Telugu inputs

---


---

## 📈 Model Architecture

- Encoder: Embedding → LSTM → Hidden States (h, c)
- Decoder: Embedding → LSTM (initial state = encoder states) → Dense (Softmax)
- Loss: Categorical Crossentropy
- Optimizer: Adam

---

