from flask import Flask, render_template

app = Flask(__name__)


# Results from our federated learning simulation
round_accuracies = [
    50.00,
    50.56,
    50.56,
    51.67,
    58.89,
    59.44,
    59.44,
    61.11,
    60.00,
    61.11
]


@app.route("/")
def home():

    final_accuracy = round_accuracies[-1]

    return render_template(
        "index.html",
        hospitals=3,
        rounds=len(round_accuracies),
        final_accuracy=final_accuracy,
        privacy="Enabled",
        accuracies=round_accuracies
    )


if __name__ == "__main__":
    app.run(debug=True)