# Research Assistant – Paper Summariser & Question Answering App  
A Streamlit-powered AI tool for reading, summarising, and querying research papers.

---

## Overview  

The **Research Assistant App** helps researchers, students, and professionals quickly understand academic papers. Instead of manually reading long PDFs, the app allows users to:

- Upload one or more research papers  
- Generate structured AI summaries  
- Ask questions based on the uploaded documents  
- Receive grounded answers with citations  

---

## Key Features  

| Feature | Description |
|--------|------------|
| Multi-PDF Upload | Supports multiple research papers at once |
| Smart Chunking | Splits papers into embedding-friendly text blocks |
| AI Summary | Short, detailed, or bullet-style summaries |
| Conversation Mode | Ask follow-up questions — works like a chatbot |
| Source Citations | Shows which section of the paper the answer came from |
| Download Transcript | Export the chat as a text file |

---

## Tech Stack  

- Python 3.10+
- Streamlit
- FAISS Vector Store
- Sentence Transformers (Embeddings)
- OpenAI Model for Retrieval-Augmented Generation
- PDFPlumber

---

## 📦 Installation  

Clone the repository:

```bash
git clone https://github.com/etiu/Research-Assistant-Paper-Summariser-Question-Answering-App.git
cd research-assistant


## Install Dependencies

```bash
pip install -r requirements.txt


## Required Setup

For this app to work, you will need to create a secrets.toml file.
The secrets.toml file will create your OpenAI API KEY.
Follow these instructions:

1. Create a folder called ".streamlit" inside your project
2. Inside this folder, create a new file called "secrets.toml"
3. Paste the following into the file and replace the placeholder text with your own OPENAI API KEY.
[api_keys]
openai="YOUR_OPENAI_API_KEY"

NOTE: This toml file must stay private and not uploaded to github.

## Run the application
On your terminal, run the app using this command:

streamlit run app2.py


The app will open in your browser at:

http://localhost:8501/

## Folder structure

Research Assistant – Paper Summariser & Question Answering App
 ┣ .streamlit
 ┃ ┗ secrets.toml
 ┣ app2.py
 ┣ requirements.txt
 ┗ README.md
