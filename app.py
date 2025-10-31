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
        return "Please type something 💕"

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
        {'<br>'.join(results)}
    </div>
    """

# 🍓 3️⃣ Custom CSS — cute, readable, and clear output box
custom_css = """
body {
    background-color: #FFE6EB;
}
.gradio-container {
    background-color: #FFE6EB !important;
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
    background-color: #ffb6c1 !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    transition: 0.3s ease;
}
button:hover {
    background-color: #ffa8b8 !important;
}

/* 🍓 Output box styling (like “Generated Response”) */
.response-box {
    background-color: #ffd6e0 !important; /* soft strawberry cream */
    border: 2px solid #ffb6c1 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    color: #4b2b30 !important;
    font-size: 16px !important;
    line-height: 1.6;
    margin-top: 10px;
    box-shadow: 2px 3px 6px rgba(255, 182, 193, 0.4);
    white-space: pre-wrap;
}
.response-label {
    font-weight: bold;
    color: #d36b83;
    font-size: 18px;
    font-family: "Poppins", sans-serif;
    margin-bottom: 5px;
    display: block;
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
    output = gr.HTML(label="Generated Response 🍓")  # visible output box label
    generate_btn = gr.Button("✨ Predict Next Words ✨")
    generate_btn.click(predict_next_words, inputs=text_input, outputs=output)
    gr.HTML("<p style='text-align:center;color:#c07085;'>Powered by 🍓 Effort</p>")

# 🍓 5️⃣ Launch
if __name__ == "__main__":
    demo.launch()
