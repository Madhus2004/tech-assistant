from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = "secret-key"


@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():

    role = request.form["role"]

    session["role"] = role

    return redirect("/chat")


@app.route("/chat")
def chat():

    role = session.get("role")

    if not role:
        return redirect("/")

    return render_template(
        "chat.html",
        role=role
    )


if __name__ == "__main__":
    app.run(debug=True)