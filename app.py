from flask import Flask, render_template, request
from model import product_recommendations_user

# load user_final_rating, df_sent_analysis, tfidf_vectorizer, rf_best_model here

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def recommend():
    error_message = None
    recommendations = None

    if request.method == "POST":
        user = request.form.get("user", "").strip()

        if not user:
            error_message = "Please enter a valid user."
        else:
            df_recommendations = product_recommendations_user(
                user_name=user
            )

            if df_recommendations is None or df_recommendations.empty:
                error_message = f"No recommendations found for user '{user}'."
            else:
                recommendations = df_recommendations.to_dict(orient="records")
                return render_template("index.html", recommendations=recommendations, user_name=user)

    return render_template(
        "index.html",
        error_message=error_message,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
