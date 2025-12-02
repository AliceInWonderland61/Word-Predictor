import html
from typing import List, Tuple

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# =========================================================
#                 Load models
# =========================================================
# Text Generation: GPT-2 (for natural text continuation)
GEN_MODEL_NAME = "gpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME)
gen_tokenizer.pad_token = gen_tokenizer.eos_token

# Text Summarization: DistilBART
SUM_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)


# =========================================================
#      Helper: collect chosen tokens + top-k alternatives
# =========================================================
def _collect_generation_tokens(
    chosen_token_ids: List[int],
    scores: List[torch.Tensor],
    tokenizer,
    top_k: int = 5,
) -> List[Tuple[str, list]]:
    """
    For each generated position (step), return:
      (chosen_token_text, [(alt_text, prob), ... up to top_k])
    We skip special tokens like </s>, <pad>, etc.
    """
    tokens_info: List[Tuple[str, list]] = []
    special_ids = set(tokenizer.all_special_ids or [])

    for chosen_id, step_scores in zip(chosen_token_ids, scores):
        # step_scores: shape (batch_size, vocab_size) - take first batch
        logits = step_scores[0]
        probs = F.softmax(logits, dim=-1)

        # Decode chosen token
        chosen_tok = tokenizer.decode([chosen_id]).strip()

        # Skip ALL special tokens
        if chosen_id in special_ids:
            continue
        
        # Skip empty tokens
        if not chosen_tok:
            continue

        # Get a surplus of top candidates, then filter
        k_surplus = max(top_k + 20, 40)
        top_vals, top_idx = torch.topk(probs, k=min(k_surplus, probs.numel()))

        alts = []
        for idx, p in zip(top_idx.tolist(), top_vals.tolist()):
            if idx == chosen_id or idx in special_ids:
                continue
            tok = tokenizer.decode([idx]).strip()
            if not tok:
                continue
            alts.append((tok, float(p)))
            if len(alts) >= top_k:
                break

        tokens_info.append((chosen_tok, alts))

    return tokens_info


# =========================================================
#           Helper: render pretty HTML output
# =========================================================
def _render_interactive_html(tokens_info, title_label: str) -> str:
    """
    Render all tokens in one line. When you hover a token, a WIDE
    docked panel at the bottom of the pink box shows 5 alternatives.
    """

    if not tokens_info:
        return f"""
        <div class="response-box">
            <div class="response-label">{html.escape(title_label)}</div>
            <div class="generated-text">(no output)</div>
        </div>
        """

    # Build spans for each token + its nested docked panel
    spans = []
    for token_text, alts in tokens_info:
        safe_tok = html.escape(token_text, quote=False)

        if alts:
            alt_html = "".join(
                f"<span class='alt-item'><b>{html.escape(t, quote=False)}</b>"
                f"<span class='pct'> ({p*100:.0f}%)</span></span>"
                for t, p in alts
            )
            dock_panel = (
                "<div class='docked-panel'>"
                f"<span class='dock-title'>Alternatives for \"{safe_tok}\":</span>"
                f"{alt_html}</div>"
            )
            span_html = f"<span class='token-span'>{safe_tok}{dock_panel}</span>"
        else:
            # Input tokens (for generation mode) - no alternatives
            span_html = f"<span class='input-token'>{safe_tok}</span>"

        spans.append(span_html)

    tokens_html = " ".join(spans)

    return f"""
    <div class="response-box">
        <div class="response-label">{html.escape(title_label)}</div>
        <div class="generated-text">{tokens_html}</div>
        <div class="instruction-footer">
            💕 Hover over a word to see the model's top alternatives 💕
        </div>
    </div>
    """


# =========================================================
#                 Inference functions
# =========================================================
def run_generation(prompt, max_tokens, temperature, top_k):
    if not prompt.strip():
        return (
            "<div class='response-box'>"
            "<div class='response-label'>⚠️ Please enter a prompt</div>"
            "</div>"
        )

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,  # greedy decoding
            repetition_penalty=1.2,  # Prevent repetitive loops
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

    # Get the full sequence
    seq = outputs.sequences[0]
    
    # Get ALL tokens (input + generated) for display
    all_token_ids = seq.tolist()
    
    # Create token info for the input tokens (no alternatives since they're given)
    tokens_info = []
    
    # Add input tokens (without alternatives)
    for token_id in all_token_ids[:input_length]:
        token_text = gen_tokenizer.decode([token_id])
        if token_text.strip():  # Only add if not empty
            tokens_info.append((token_text, []))  # Empty list = no alternatives
    
    # Add generated tokens (with alternatives)
    generated_ids = all_token_ids[input_length:]
    generated_info = _collect_generation_tokens(
        generated_ids,
        outputs.scores,
        gen_tokenizer,
        top_k=int(top_k),
    )
    
    tokens_info.extend(generated_info)

    return _render_interactive_html(tokens_info, "🌸 Generated Output")


def run_summarization(text, max_tokens, top_k):
    if not text.strip():
        return (
            "<div class='response-box'>"
            "<div class='response-label'>⚠️ Please enter text</div>"
            "</div>"
        )

    inputs = sum_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = sum_model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            num_beams=4,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = outputs.sequences[0]
    # Skip the first token (decoder start token)
    chosen_ids = seq[1 : 1 + len(outputs.scores)].tolist()

    tokens_info = _collect_generation_tokens(
        chosen_ids, outputs.scores, sum_tokenizer, top_k=int(top_k)
    )

    return _render_interactive_html(tokens_info, "🍰 Summary")


# =========================================================
#                        CSS
# =========================================================
custom_css = """
/* -------- Overall theme (UI A style) -------- */
:root, [data-theme="dark"] {
    --color-text: #fbe7f1 !important;
    --color-text-secondary: #f9c3da !important;
    --color-background-primary: #1e1e22 !important;
    --color-background-secondary: #1e1e22 !important;
    --block-background-fill: transparent !important;
}
/* Soft striped background */
body {
    background: repeating-linear-gradient(
        90deg,
        #ffeef7 0px,
        #ffeef7 120px,
        #ffe0f0 120px,
        #ffe0f0 240px
    ) !important;
}
/* Center app */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
/* Headings */
h1, h2, h3, p {
    color: #d946a6 !important;  /* Darker pink for better readability */
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;  /* Subtle shadow for depth */
}
/* Dark rounded cards */
.card {
    background: #18181b !important;
    border-radius: 24px !important;
    border: 2px solid #ffb6d9 !important;
    padding: 22px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
}
/* Text areas / inputs */
textarea, input[type="text"] {
    background: #ffe7f3 !important;
    border-radius: 14px !important;
    border: 2px solid #f3b6cc !important;
    color: #2d111b !important;
}
/* Labels */
.label, .gr-label, .label-wrap, label {
    color: #ffe7f1 !important;
    font-weight: 600 !important;
}
/* ------- Heart radio buttons ------- */
input[type="radio"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 22px !important;
    height: 22px !important;
    margin-right: 8px !important;
    cursor: pointer !important;
    position: relative !important;
    outline: none !important;
}
/* Empty heart */
input[type="radio"]::before {
    content: "♡";
    position: absolute;
    font-size: 22px;
    color: #ffb6d9;
    top: -3px;
    left: -1px;
    transition: all 0.25s ease;
}
/* Filled heart when selected */
input[type="radio"]:checked::before {
    content: "❤";
    color: #ff6fa3;
    transform: scale(1.15);
    text-shadow: 0 0 6px rgba(255,111,163,0.7);
}
/* Chip around the label text */
input[type="radio"] + label {
    background: #242428 !important;
    border-radius: 999px !important;
    border: 1px solid #ffb6d9 !important;
    padding: 6px 20px 6px 36px !important;
    color: #ffe7f1 !important;
    cursor: pointer !important;
    transition: all 0.18s ease;
}
input[type="radio"] + label:hover {
    background: #2f2f34 !important;
}
input[type="radio"]:checked + label {
    background: #ffb6d9 !important;
    color: #2b1018 !important;
    box-shadow: 0 4px 14px rgba(255,182,217,0.6);
}
/* ------- Run button ------- */
button.primary {
    background: linear-gradient(135deg, #ffb6d9, #ff8fbf) !important;
    border-radius: 32px !important;
    border: none !important;
    color: #2b1018 !important;
    font-weight: 800 !important;
    padding: 16px 28px !important;
    width: 100% !important;
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    transition: transform 0.12s ease, box-shadow 0.2s ease;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(0,0,0,0.45);
}
button.primary:active {
    transform: scale(0.96);
    box-shadow: 0 8px 18px rgba(0,0,0,0.4);
}
/* ------- Response / tokens area ------- */
.response-box {
    background: #ffe9f2 !important;
    border-radius: 22px;
    border: 2px solid #f3b6cc;
    padding: 16px 20px 80px 20px; /* extra space for docked panel */
    margin-top: 18px;
    position: relative;           /* docked-panel anchors here */
    box-shadow: 0 8px 18px rgba(0,0,0,0.15);
}
.response-label {
    font-size: 18px;
    font-weight: 700;
    color: #d26b93;
    border-bottom: 2px dashed #f3b6cc;
    padding-bottom: 6px;
    margin-bottom: 8px;
}
/* Make sure text is visible (fix for white text issue) */
.generated-text,
.generated-text *,
.token-span,
.token-span *,
.input-token {
    color: #2d111b !important;
    font-weight: 600;
    font-size: 16px;
}
/* Input tokens (no underline) */
.input-token {
    padding: 1px 2px;
}
/* Underlined tokens (generated) */
.token-span {
    border-bottom: 1px solid rgba(233,139,177,0.5);
    padding: 1px 2px;
    cursor: pointer;
}
.token-span:hover {
    background: #ffd7ec;
}
/* Wide docked panel at bottom of the box */
.docked-panel {
    display: none;
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    background: #ffe4f0;
    border-top: 2px solid #f3b6cc;
    padding: 10px 16px;
    box-sizing: border-box;
    overflow-x: auto;
    white-space: nowrap;
}
/* Show the panel when hovering the token */
.token-span:hover .docked-panel {
    display: flex;
    align-items: center;
    gap: 10px;
}
/* Alternative chips */
.alt-item {
    background: #ffffff;
    border-radius: 999px;
    border: 2px solid #e7b8cb;
    padding: 5px 12px;
    font-size: 13px;
    white-space: nowrap;
}
.alt-item b {
    color: #c65485;
}
.pct {
    margin-left: 4px;
    font-size: 0.83em;
    color: #6b5e63;
}
.dock-title {
    font-weight: 700;
    margin-right: 10px;
    color: #c65485;
}
/* Footer hint text */
.instruction-footer {
    position: absolute;
    bottom: 4px;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 13px;
    color: #d26b93;
    transition: opacity 0.2s ease;
    pointer-events: none; /* Don't block clicks */
}
/* Hide instruction when hovering any token */
.token-span:hover ~ .instruction-footer,
.response-box:has(.token-span:hover) .instruction-footer {
    opacity: 0;
}
"""


# =========================================================
#                     UI
# =========================================================
with gr.Blocks(css=custom_css, title="💕 Word Predictor — Text Generation & Summarization") as demo:
    gr.Markdown(
        "## 💕 Word Predictor — Text Generation & Summarization\n"
        "Single text field with a mode toggle. **Hover any word** in the output to see 5 alternative tokens."
    )

    # Top card: mode + text input
    with gr.Column(elem_classes="card"):
        mode = gr.Radio(
            ["Text Generation", "Text Summarization"],
            label="Mode",
            value="Text Generation",
        )
        text_input = gr.Textbox(
            lines=5,
            label="Enter your text / prompt:",
            placeholder="Type your prompt here…",
        )

    # Second card: sliders
    with gr.Column(elem_classes="card"):
        max_tokens = gr.Slider(
            10, 300, value=50, step=10, label="Max New Tokens"
        )
        temperature = gr.Slider(
            0.1, 1.5, value=1.0, step=0.1, label="Temperature"
        )
        topk = gr.Slider(
            1, 10, value=5, step=1, label="Top-k Alternatives"
        )

    output_html = gr.HTML()
    run_btn = gr.Button("Run", variant="primary")

    # Decide which function to run based on mode
    def run_mode(mode_value, txt, max_t, temp, k):
        if mode_value == "Text Generation":
            return run_generation(txt, max_t, temp, k)
        else:
            # summarization ignores temperature (standard beam search)
            return run_summarization(txt, max_t, k)

    run_btn.click(
        run_mode,
        inputs=[mode, text_input, max_tokens, temperature, topk],
        outputs=output_html,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )