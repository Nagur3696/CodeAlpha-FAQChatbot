from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ data
faq = pd.read_csv("faq.csv")

questions = faq["Question"].tolist()
answers = faq["Answer"].tolist()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""

    if request.method == "POST":
        user_question = request.form["question"]

        user_vector = vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, question_vectors)
        index = similarity.argmax()

        if similarity[0][index] > 0.2:
            response = answers[index]
        else:
            response = "Sorry, I don't know the answer."

    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True)