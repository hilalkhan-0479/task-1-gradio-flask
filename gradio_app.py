import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import DATASETS, get_model


def info(dataset):
    m = get_model(dataset)
    summary = (
        f"**Target:** {m.target}\n\n"
        f"**Features:** {', '.join(m.features)}\n\n"
        f"**Metrics:** " + " | ".join(f"{k}: {v}" for k, v in m.metrics.items()) + "\n\n"
        f"**Intercept:** {m.intercept}\n\n"
        f"**Coefficients:** " + ", ".join(f"{k}: {v}" for k, v in m.coefs.items())
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    X, y = m.df[m.features], m.df[m.target]
    ax.scatter(y, m.model.predict(X), alpha=0.7)
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    return summary, m.df.head(10), fig


def predict(dataset, *vals):
    m = get_model(dataset)
    vals = vals[: len(m.features)]
    if any(v is None for v in vals):
        return "Please fill in all inputs."
    return f"Predicted {m.target}: {m.predict(list(vals)):,.2f}"


def build_inputs(dataset):
    """Show only as many number boxes as the dataset has features (max 5)."""
    m = get_model(dataset)
    rngs = m.feature_ranges()
    updates = []
    for i in range(5):
        if i < len(m.features):
            f = m.features[i]
            lo, hi, med = rngs[f]
            updates.append(gr.update(visible=True, label=f"{f} (range {lo:g}–{hi:g})", value=med))
        else:
            updates.append(gr.update(visible=False, value=None))
    return updates


with gr.Blocks(title="Linear Regression - Gradio") as demo:
    gr.Markdown("# Linear Regression Predictor (Gradio)")
    ds = gr.Dropdown(list(DATASETS), value=list(DATASETS)[0], label="Choose dataset")
    with gr.Row():
        with gr.Column():
            info_md = gr.Markdown()
            table = gr.Dataframe(label="Dataset preview (first 10 rows)")
        plot = gr.Plot()
    gr.Markdown("### Make a prediction")
    inputs = [gr.Number(visible=False) for _ in range(5)]
    btn = gr.Button("Predict", variant="primary")
    out = gr.Textbox(label="Result")

    ds.change(info, ds, [info_md, table, plot])
    ds.change(build_inputs, ds, inputs)
    demo.load(info, ds, [info_md, table, plot])
    demo.load(build_inputs, ds, inputs)
    btn.click(predict, [ds] + inputs, out)

if __name__ == "__main__":
    demo.launch()
