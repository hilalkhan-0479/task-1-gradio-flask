from flask import Flask, render_template, request
from model import DATASETS, get_model

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    name = request.values.get("dataset", list(DATASETS)[0])
    m = get_model(name)
    result, error, entered = None, None, {}
    if request.method == "POST" and request.form.get("action") == "predict":
        try:
            entered = {f: float(request.form[f]) for f in m.features}
            result = f"{m.predict(list(entered.values())):,.2f}"
        except (ValueError, KeyError):
            error = "Please enter valid numbers in all fields."
    return render_template(
        "index.html",
        datasets=list(DATASETS), name=name, m=m,
        ranges=m.feature_ranges(), preview=m.df.head(10),
        result=result, error=error, entered=entered,
    )


if __name__ == "__main__":
    app.run(debug=True)
