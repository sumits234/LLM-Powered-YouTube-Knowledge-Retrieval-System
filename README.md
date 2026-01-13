<div align="center">

<img src="assets/logo.png" width="130" alt="Project Logo"/>

# 🎥 LLM-Powered YouTube Knowledge Retrieval System
### Ask questions from any YouTube video like ChatGPT (RAG + Transcript + Vector Search)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange.svg)](https://faiss.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

🔗 **GitHub Repo:** https://github.com/sumits234/LLM-Powered-YouTube-Knowledge-Retrieval-System

</div>

---

## 🚀 Project Overview
This project is a **YouTube RAG (Retrieval Augmented Generation) system** that allows users to:
- Enter a **YouTube video URL**
- Automatically fetch **video transcript**
- Convert transcript into embeddings
- Store in a **Vector Database**
- Ask questions and get **accurate contextual answers powered by an LLM**

✅ Works like *ChatGPT for YouTube videos*.

---

## ✨ Features
✅ Extract transcripts from YouTube videos  
✅ Chunk transcript + create embeddings  
✅ Store embeddings using **FAISS** vector DB  
✅ Context-based Q&A using **RAG pipeline**  
✅ Streamlit UI for smooth interaction  
✅ Accurate answers with source context  

---

## 🧠 Tech Stack
- **Python**
- **LangChain**
- **FAISS (Vector DB)**
- **OpenAI / LLM**
- **Streamlit**
- **YouTube Transcript API**

---

## ⚙️ System Architecture (Flow)
```txt
YouTube URL → Transcript Extraction → Text Chunking
          → Embeddings Generation → FAISS Vector Store
          → User Query → Similarity Search → LLM Answer
