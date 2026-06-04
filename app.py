# app.py
import os
from flask import (Flask, render_template, request,
                   session, redirect, url_for, jsonify)
from dotenv import load_dotenv
from src.rbac import authenticate_user, get_role_badge_color
from src.retriever import retrieve_documents, format_context
from src.prompt import build_prompt, SYSTEM_PROMPT
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret-key")

# Initialize Groq LLM once at startup
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,        # low temperature = factual, consistent answers
    max_tokens=1024
)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Redirect root to login or chat depending on session."""
    if "user" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page — validates credentials and sets session."""
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        user = authenticate_user(username, password)

        if user:
            session["user"] = user
            return redirect(url_for("chat"))
        else:
            error = "Invalid username or password. Please try again."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Clears session and redirects to login."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat", methods=["GET"])
def chat():
    """Main chat interface — requires login."""
    if "user" not in session:
        return redirect(url_for("login"))

    user = session["user"]
    badge_color = get_role_badge_color(user["role"])
    return render_template("chat.html", user=user, badge_color=badge_color)


@app.route("/ask", methods=["POST"])
def ask():
    """
    API endpoint that handles a chat question.

    Flow:
    1. Validate session
    2. Get role from session
    3. Retrieve relevant chunks from Pinecone (RBAC filtered)
    4. Build prompt with context
    5. Send to Groq LLM
    6. Return answer + sources
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    user = session["user"]
    role = user["role"]

    try:
        # Step 1: Retrieve relevant documents (RBAC enforced here)
        documents = retrieve_documents(query=question, role=role, top_k=4)

        # Step 2: Format context for LLM
        context = format_context(documents)

        # Step 3: Build prompt
        prompt = build_prompt(
            question=question,
            context=context,
            role=role
        )

        # Step 4: Call Groq LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        answer = response.content

        # Step 5: Extract unique sources for citation display
        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in documents
        })

        return jsonify({
            "answer": answer,
            "sources": sources,
            "role": role,
            "namespaces_searched": list({
                doc.metadata.get("role", "unknown")
                for doc in documents
            })
        })

    except Exception as e:
        print(f"Error in /ask: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)