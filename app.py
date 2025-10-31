import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 🍓 1️⃣ Load model + tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 🍓 2️⃣ Function that predicts top 5 next words for each token
def predict_next_words(text):
    if not text.strip():
        return """
        <div class='response-box'>
            <span class='response-label'>Predicted Next Tokens 🍓</span>
            Type something above to see predictions 💕
        </div>
        """

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    results = []

    for i in range(len(input_ids)):
        prefix = input_ids[:i + 1].unsqueeze(0)
        with torch.no_grad():
            outputs = model(prefix)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=5)
            top_tokens = [tokenizer.decode([tid]).strip() for tid in topk.indices[0]]

        current_token = tokenizer.decode([input_ids[i]])
        current_token = current_token if current_token.strip() else "(space)"
        results.append(f"**{current_token}** → {', '.join(top_tokens)}")

    # ✨ Return output wrapped in a styled div
    return f"""
    <div class='response-box'>
        <span class='response-label'>Predicted Next Tokens 🍓</span>
        {'<br>'.join(results)}
    </div>
    """

# 🍓 3️⃣ Custom CSS — cute, readable, and clear output box
custom_css = """
body {
    background: linear-gradient(180deg, #fff8fa 0%, #ffe6eb 35%, #ffd6e0 100%);
    background-attachment: fixed;
}
.gradio-container {
    background: transparent !important;
}
textarea, input {
    background-color: #fffdfd !important;
    color: #5a3d3d !important;
    border-radius: 20px !important;
    border: 2px solid #ffc9d6 !important;
    font-size: 16px !important;
    padding: 8px !important;
}
button {
    background: linear-gradient(90deg, #ffb6c1, #ff9eb0);
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    transition: 0.3s ease;
}
button:hover {
    background: linear-gradient(90deg, #ff9eb0, #ffb6c1);
}

/* 🍓 Output box styling (persistent) */
.response-box {
    background-color: rgba(255, 230, 235, 0.9);
    border: 2px solid #ffb6c1;
    border-radius: 15px;
    padding: 20px;
    color: #4b2b30;
    font-size: 16px;
    line-height: 1.6;
    margin-top: 15px;
    box-shadow: 3px 4px 8px rgba(255, 182, 193, 0.3);
    white-space: pre-wrap;
    min-height: 150px;
    width: 100%;
}
.response-label {
    font-weight: bold;
    color: #d36b83;
    font-size: 18px;
    font-family: "Poppins", sans-serif;
    margin-bottom: 8px;
    display: block;
    border-bottom: 1.5px dashed #ffc9d6;
    padding-bottom: 5px;
    text-align: center;
}
/* 🍰 Force the output box to always show */
#output-box {
    display: block !important;
    width: 100%;
}
h1, h3 {
    color: #d36b83;
    font-family: "Comic Sans MS", "Poppins", sans-serif;
    text-align: center;
}
"""

# 🍓 4️⃣ Build the Gradio UI
demo = gr.Blocks(css=custom_css)

with demo:
    gr.HTML("<h1>🍓 Word Predictor 🍓</h1><h3>Type a sweet sentence and see what the AI thinks comes next!</h3>")
    text_input = gr.Textbox(
        placeholder="Type your text here... 🍰",
        label="Your Text 💌"
    )

    with gr.Row():
        output = gr.HTML(
            value="""
            <div class='response-box'>
                <span class='response-label'>Predicted Next Tokens 🍓</span>
                Type something above to see predictions 💕
            </div>
            """,
            elem_id="output-box",
            label="Generated Response 🍓",
            show_label=False
        )

    generate_btn = gr.Button("✨ Predict Next Words ✨")
    generate_btn.click(predict_next_words, inputs=text_input, outputs=output)
    gr.HTML("<p style='text-align:center;color:#c07085;'>Powered by 🍓 Effort</p>")

# 🍓 5️⃣ Launch
if __name__ == "__main__":
    demo.launch()
