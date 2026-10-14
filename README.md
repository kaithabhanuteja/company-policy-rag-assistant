# 🤖 Company Policy RAG Assistant

An AI-powered Company Policy Assistant that helps employees quickly get answers to questions about company policies such as **travel, expenses, leave, working hours, benefits, and other HR-related topics**.

Instead of HR and Operations teams repeatedly answering the same questions, employees can ask the AI assistant and get answers based on the company's policy document.

## 🎯 Problem Statement

HR and Operations teams spend significant time answering repetitive employee questions about company policies, travel, expenses, leave, reimbursements, and other procedures.

This project uses **Retrieval-Augmented Generation (RAG)** to automatically find relevant information from company policy documents and generate useful answers.

## 💡 Solution

The system works in three main steps:

**Employee Question → Retrieve Relevant Policy Information → Generate Answer**

The system searches the company policy document for relevant information using vector similarity and then uses the Gemini API to generate a clear response.

## ⚙️ How It Works

1. The company policy PDF is uploaded to the system.
2. The PDF text is extracted using **PyMuPDF**.
3. The document is divided into smaller chunks.
4. **Sentence Transformers** converts the chunks into embeddings.
5. **FAISS** stores and searches these embeddings.
6. When an employee asks a question, the system finds the most relevant policy sections.
7. The retrieved information is provided to **Google Gemini**.
8. Gemini generates the final answer.
9. The answer is displayed through a simple **Flask web application**.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend and web interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Google Gemini API | AI answer generation |
| HTML/CSS | Frontend |
| python-dotenv | Environment variable management |

## 🧠 AI Components

### Sentence Transformers

Converts the policy document and user questions into numerical vectors called **embeddings**.

### FAISS

Searches the embeddings to find the policy sections that are most relevant to the user's question.

### Google Gemini

Uses the retrieved policy information to generate a natural-language answer.

## 📂 Project Structure

```text
company_policy_rag/
│
├── app.py
├── main.py
├── company_policy.pdf
├── requirements.txt
├── .env.example
├── .gitignore
│
└── templates/
    └── index.html
