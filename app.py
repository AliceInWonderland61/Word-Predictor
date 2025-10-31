import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 1️⃣ Load model + tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2️⃣ Define prediction function
def predict_next_words(text):
    if not text.strip():
        return "Please type something 💕"

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    results = []
    # Loop through tokens in user input
    for i in range(len(input_ids)):
        prefix = input_ids[:i+1].unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            outputs = model(prefix)
            logits = outputs.logits[:, -1, :]   # last token's logits
            probs = F.softmax(logits, dim=-1)
            topk = torch.topk(probs, k=5)
            top_tokens = [tokenizer.decode([tid]).strip() for tid in topk.indices[0]]

        current_token = tokenizer.decode([input_ids[i]])
        # Handle empty tokens (like punctuation)
        current_token = current_token if current_token.strip() else "(space)"
        results.append(f"**{current_token}** → {', '.join(top_tokens)}")

    return "\n".join(results)

# 3️⃣ Custom CSS styling
custom_css = """
body {background-color: #FFE6EB;}
.gradio-container {background-color: #FFE6EB !important;}
textarea, input {
    background-color: #fff8f9 !important;
    border-radius: 20px !important;
    border: 2px solid #ffc9d6 !important;
}
button {
    background-color: #ffb6c1 !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
}
button:hover {
    background-color: #ffa8b8 !important;
}
.output-markdown {
    background-color: #fff0f5 !important;
    border-radius: 15px !important;
    padding: 10px;
}
h1, h3 {
    color: #d36b83;
    font-family: "Comic Sans MS", "Poppins", sans-serif;
    text-align: center;
}
"""  # ✅ <--- This closing triple quote is REQUIRED
