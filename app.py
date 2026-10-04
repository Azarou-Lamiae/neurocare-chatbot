"""Flask entry point for the NeuroCare chatbot."""
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from neurocare.chatbot import handle_user_input

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static",
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    user_text = data.get("message")

    if not user_text:
        return jsonify({"response": "Empty message."}), 400

    return jsonify({"response": handle_user_input(user_text)})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        port=int(os.getenv("PORT", "5000")),
    )
