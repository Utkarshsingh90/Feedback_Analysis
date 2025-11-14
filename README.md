# 🚔 AI Feedback Analytics Platform

**Intelligent Text Analysis & Insights Generation System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

A Streamlit-powered web application that transforms unstructured feedback into actionable insights using state-of-the-art AI/ML models.

---
## 📋 Table of Contents
- Overview
- Features
- AI/ML Models
- Tech Stack
- Installation
- Usage
- Configuration
- Project Structure
- API Reference
- Screenshots
- Model Working
- Future Improvements

---

## 🎯 Overview
The **AI Feedback Analytics Platform** processes unstructured text from feedback, news, and social media. It uses machine learning models and extractors to provide insights like sentiment analysis, entity extraction, and competency tagging.

---

## ✨ Features
### 🤖 AI-Powered Analysis
- Sentiment classification
- Summarization
- Question-answering
- Language detection

### 📊 Custom Intelligence
- Entity extraction
- Competency tagging
- Recognition scoring

### 🎨 Dashboard
- Real-time analytics
- Bar charts
- Recent submissions table

### 💾 Export Options
- PDF
- CSV & JSON

---

## 🧠 AI/ML Models
Uses DistilBERT, BART, and RoBERTa for sentiment, summarization, and Q&A.

Recognition Score = min(1.0, Base_Score + Tag_Boost + Length_Boost)

---

## 🛠️ Tech Stack
- Streamlit
- Transformers
- PyTorch
- Langdetect
- Pandas
- ReportLab
- pdfplumber

---

## 📦 Installation
Clone repo, create venv, install dependencies, run Streamlit app.

```bash
git clone https://github.com/yourusername/ai-feedback-analytics.git
cd ai-feedback-analytics
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Usage
Supports text input, file upload, dashboard insights, Q&A, and export.

---

## ⚙️ Configuration
Modify districts, departments, competency tags, and model parameters in `app.py`.

---

## 📁 Project Structure
```
ai-feedback-analytics/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── assets/
├── data/
├── tests/
└── docs/
```
---

## 📸 Screenshots

### Feedback Form
![Feedback Form](Feedback_form.png)

### Analysis Results
![Analysis Results](Analysis_Results.png)

### Dashboard
![Dashboard](Dashboard.png)

### Export Data Page
![Export Data](Export_data.png)

### Q&A Chat
![Q&A Chat](Q&A_chat.png)

---

## 🔌 API Reference
Includes `load_models()`, `process_text()`, `extract_simple_entities()`, and `calculate_recognition_score()`.

---

## 📞 Contact
Email: 22cs01075@iitbbs.ac.in

---

## ➕ Additional Files
### requirements.txt
```
streamlit>=1.28.0
transformers>=4.30.0
torch>=2.0.0
pandas>=2.0.0
langdetect>=1.0.9
reportlab>=4.0.0
pdfplumber>=0.10.0
```

### .gitignore
```
__pycache__/
*.py[cod]
venv/
.streamlit/
transformers_cache/
```
## AI Models Working


### 1. How the AI/ML Models Are Used
You are using three pre-trained "pipeline" models from Hugging Face Transformers.


#### **Sentiment Analysis**
- **Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Purpose:** Reads the full feedback and decides whether the sentiment is *positive* or *negative*.
- **Used for:**
- Showing the sentiment emoji (😊 or 😞)
- Calculating the **Recognition Score** (positive → higher score)


#### **Summarization**
- **Model:** facebook/bart-large-cnn
- **Purpose:** Reads the complete text and generates a shorter paragraph capturing the key points.
- **Used for:**
- The **Summary** box in the "View Details" section
- Preview summaries in the **Dashboard** tab


#### **Question-Answering (Q&A)**
- **Model:** deepset/roberta-base-squad2
- **Purpose:** Powers the **Q&A Chat** tab by answering user questions based on all collected feedback.
- **Inputs:**
- A question from the user
- A combined context made from all processed feedback
- **Process:** Extracts the exact span of text from the context that answers the question.


---


### 2. How They Work (Simple Explanation)


#### **Sentiment Analysis (DistilBERT)**
A text classifier trained on thousands of reviews. It learned patterns like:
- "excellent", "appreciated", "helpful" → Positive
- "poor", "rude", "disappointed" → Negative


It reads your entire feedback and predicts which label fits best.


#### **Summarization (BART)**
A sequence-to-sequence model with two components:
- **Encoder:** Reads and understands your full text
- **Decoder:** Generates a brand-new shorter version, word by word


Works like a smart journalist who rewrites long paragraphs into concise summaries.


#### **Question-Answering (RoBERTa)**
An extractive model — it doesn’t invent answers. Instead, it:
- Looks at the question
- Searches the context
- Highlights the exact words that answer the question
```
```
## 🔮 Future Improvements
Your current AI system includes two advanced ML components (sentiment, summarization, Q&A) and several simple keyword-based components (districts, departments, competencies). The most impactful upgrades come from improving the latter.


### **1. Use Zero‑Shot Classification for Competency Tags**
**Problem:** `extract_competency_tags()` only finds tags if exact keywords appear in text. It misses cases like:
> "The officer was kind and understanding" → should match **compassion**, but keyword isn't present.


**Solution:** Use a **Zero‑Shot Classification** model.
- This model understands the *meaning* of text and labels, even without exact keywords.
- Provide feedback + your labels:
- `bravery`, `professionalism`, `compassion`, `life saving`, `rapid response`, etc.
- Model returns the best‑matching competencies with confidence scores.


**Benefit:** Much smarter, semantic tag extraction.


---


### **2. Use a Named Entity Recognition (NER) Model for Districts, Departments & Officer Names**
**Problem:** `extract_simple_entities()` depends on fixed keyword lists.
- Cannot detect new department names.
- Cannot detect officer names (e.g., *“Constable Das”*).


**Solution:** Add a pre-trained **NER** pipeline.
- Identifies `PERSON`, `ORG`, and `LOCATION` entities.
- Extracts:
- Police officer names
- Station/department names
- Districts and localities


**Benefit:** More accurate and expandable entity extraction.


---


### **3. Implement RAG (Retrieval‑Augmented Generation) for Smart Q&A**
**Problem:** Current Q&A loads *all feedback text* into a single context.
- Slow
- Breaks with large datasets (token limit)
- Not scalable


**Solution:** Use **RAG architecture**:
1. Store each feedback entry as a **vector embedding**.
2. Use a vector database (FAISS, Chroma, Pinecone).
3. When user asks a question:
- Retrieve top 3–5 most relevant feedback entries.
4. Feed only those into the Q&A model.


**Benefit:**
- Faster Q&A
- More accurate answers
- Scales to thousands or millions of entries

---
```

