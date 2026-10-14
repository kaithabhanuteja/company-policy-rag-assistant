
import os

from flask import Flask, render_template, request, session
from main import (
    extract_pdf,
    create_chunks,
    create_vector_store,
    answer_question,
    PDF_PATH,
    SentenceTransformer,
)

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "dev-only-change-this-secret"
)

# Initialize the RAG model once at startup.
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading company policy...")

pages = extract_pdf(PDF_PATH)
chunks = create_chunks(pages)
index = create_vector_store(chunks, embedding_model)

print("RAG knowledge base is ready!")


@app.route("/", methods=["GET", "POST"])
def home():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            try:
                answer = answer_question(
                    question,
                    chunks,
                    index,
                    embedding_model
                )
            except Exception as error:
                import traceback

                print("\n========== CHAT ERROR ==========")
                traceback.print_exc()
                print("================================\n")

                answer = f"Error: {type(error).__name__}: {error}"

            messages = session.get("messages", [])
            messages.append({
                "role": "user",
                "text": question
            })
            messages.append({
                "role": "assistant",
                "text": answer
            })

            # Keep the conversation small.
            session["messages"] = messages[-20:]
            session.modified = True

    return render_template(
        "index.html",
        messages=session.get("messages", [])
    )


@app.route("/clear", methods=["POST"])
def clear_chat():
    session["messages"] = []
    session.modified = True

    return render_template(
        "index.html",
        messages=[]
    )


if __name__ == "__main__":
    app.run(debug=True)
