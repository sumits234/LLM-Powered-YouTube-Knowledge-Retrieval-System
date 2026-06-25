<div align="center">



# 🎥 LLM-Powered YouTube Knowledge Retrieval System
### Ask questions from any YouTube video like ChatGPT (RAG + Transcript + Vector Search)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

🔗 **GitHub Repo:** https://github.com/sumits234/LLM-Powered-YouTube-Knowledge-Retrieval-System

</div>

---

## 🚀 Project Overview
This project is a **YouTube RAG (Retrieval Augmented Generation) system** that allows users to:
- Enter a **YouTube video URL**
- Automatically fetch **video transcript**
- Convert transcript into embeddings
- Store embeddings in **ChromaDB Vector Database**
- Ask questions and get **accurate contextual answers powered by an LLM**

✅ Works like *ChatGPT for YouTube videos*.

---

## ✨ Features
✅ Extract transcripts from YouTube videos  
✅ Chunk transcript + create embeddings  
✅ Store embeddings using **ChromaDB Vector DB**  
✅ Context-based Q&A using **RAG pipeline**  
✅ Streamlit UI for smooth interaction  
✅ Accurate answers with source context  

---

## 🧠 Tech Stack
- **Python**
- **Streamlit**
- **LangChain**
- **ChromaDB (Vector Database)**
- **LLM**
- **YouTube Transcript API**

---

## ⚙️ System Architecture (Flow)
```txt
YouTube URL → Transcript Extraction → Text Chunking
          → Embeddings Generation → ChromaDB Vector Store
          → User Query → Similarity Search → LLM Answer
