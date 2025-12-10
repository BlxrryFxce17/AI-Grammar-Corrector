import streamlit as st
import pandas as pd
import numpy as np
import torch
import difflib
import evaluate
from nltk.metrics import edit_distance
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from datetime import datetime
import warnings
from transformers.utils import logging
import altair as alt

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ---------------------------------------
# Page & Theme Styling
# ---------------------------------------
st.set_page_config(page_title="Grammar AI Suite", page_icon="🪶", layout="wide")
st.markdown("""
<style>
body { background-color: #f6f9fc; color: #2c3e50; }
h1, h2, h3 { color: #1a5276; }
section.main > div { border-radius: 10px; background: #ffffff; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Cache Model Load (for fast startup!)
# ---------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    base_model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, "./grammar_model_lora")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ---------------------------------------
# Correction Function – TWEAKED for best results
# ---------------------------------------
def fix_grammar(sentence, max_tokens=150):
    input_text = (
        sentence
    )
    input_ids = tokenizer(
        input_text,
        return_tensors="pt", truncation=True, padding="max_length", max_length=128
    ).input_ids.to(device)
    # TIP: Balanced between precision and creativity
    output_ids = model.base_model.generate(
        input_ids,
        max_length=max_tokens,
        num_beams=8,             # more beams for better candidates!
        do_sample=True,          # enables some creative expansion
        top_p=0.94,              # good top-p for rewriting
        temperature=0.7,         # more confident, less random
        no_repeat_ngram_size=3   # prevent phrase repetition
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# ---------------------------------------
# Explain/Linguistic Change Function
# ---------------------------------------
def explain_changes(original, corrected):
    diff = difflib.ndiff(original.split(), corrected.split())
    added = [w[2:] for w in diff if w.startswith('+ ')]
    removed = [w[2:] for w in diff if w.startswith('- ')]
    return (
        f"**Added words:** {', '.join(added) if added else 'None'}\n"
        f"**Removed words:** {', '.join(removed) if removed else 'None'}"
    )

def highlight_changes(original, corrected):
    diff = difflib.ndiff(original.split(), corrected.split())
    html_output = " ".join([
        f"<span style='color:#d9534f;text-decoration:line-through'>{w[2:]}</span>" if w.startswith('- ') else
        f"<span style='color:#28a745;font-weight:600'>{w[2:]}</span>" if w.startswith('+ ') else
        w[2:]
        for w in diff if not w.startswith('? ')
    ])
    return f"<div style='font-size:16px;line-height:1.6'>{html_output}</div>"

# ---------------------------------------
# Sidebar Navigation
# ---------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🪶 Grammar Correction", "📊 Evaluation Dashboard"])
max_tokens = st.sidebar.slider("Max output tokens", min_value=50, max_value=256, value=150)
st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.caption(f"Fine‑tuned T5‑Small + LoRA • Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
st.sidebar.caption(f"Loaded: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------------------
# Page 1: Grammar Correction
# ---------------------------------------
if page == "🪶 Grammar Correction":
    st.markdown("<h1 style='text-align:center;'>🪶 Grammar Correction AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Write/paste any sentence. Get fluent corrections and actual AI explanations below!</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✏️ Input Text")
        text_input = st.text_area("Enter your sentence…", height=250, placeholder="Type or paste text here…")
    with col2:
        st.subheader("✅ Corrected Text")
        result_box = st.empty()

    colA, colB, _ = st.columns([1, 1, 2])
    with colA:
        generate = st.button("✨ Fix Grammar", key="fix", help="Run the model correction")
    with colB:
        clear = st.button("🗑️ Clear", key="clear", help="Reset output and input")

    if generate and text_input.strip():
        with st.spinner("Correcting grammar…"):
            corrected = fix_grammar(text_input, max_tokens=max_tokens)
            st.session_state["corrected"] = corrected
            result_box.text_area("Output", value=corrected, height=250)
            st.markdown("### 🔍 Differences")
            st.markdown(highlight_changes(text_input, corrected), unsafe_allow_html=True)
            st.markdown("### 🧠 AI Change Explanation")
            st.markdown(explain_changes(text_input, corrected))
            st.download_button("💾 Download corrected text", corrected, "corrected.txt", "text/plain")
    elif clear:
        st.session_state["corrected"] = ""
        st.rerun()
    elif "corrected" in st.session_state:
        result_box.text_area("Output", value=st.session_state["corrected"], height=250)

    st.subheader("💡 Try Examples:")
    examples = [
        "He go to school every day.",
        "The report have been submitted yesterday.",
        "She don’t likes watch movie at night.",
        "I go walk everyday to work."
    ]
    example_cols = st.columns(len(examples))
    for i, e in enumerate(examples):
        if example_cols[i].button(e):
            st.session_state["corrected"] = fix_grammar(e)
            st.rerun()

# ---------------------------------------
# Page 2: Evaluation Dashboard with User Data Upload, Graphs, & Explanation
# ---------------------------------------
elif page == "📊 Evaluation Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Evaluation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload data, view metrics and professional graphs.</p>", unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload custom test data (CSV: columns Input,Expected)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        default_data = [
            {"Input": "She don't likes pizza.", "Expected": "She doesn’t like pizza."},
            {"Input": "He go to school every day.", "Expected": "He goes to school every day."},
            {"Input": "I go walk everyday to work.", "Expected": "I go for a walk every day to work."},
            {"Input": "The report have been submitted yesterday.", "Expected": "The report was submitted yesterday."}
        ]
        data = pd.DataFrame(default_data)

    st.write("### 🧾 Test Data")
    st.dataframe(data, use_container_width=True)

    run_eval = st.button("🚀 Run Evaluation")
    if run_eval:
        with st.spinner("Generating predictions and computing metrics…"):
            # Add model output and progress bar
            predictions = []
            progress = st.progress(0)
            for i, row in enumerate(data["Input"]):
                predictions.append(fix_grammar(row))
                progress.progress((i+1) / len(data))
            data["Model_Output"] = predictions

            references = data["Expected"].tolist()

            # Metrics
            rouge = evaluate.load("rouge")
            bleu = evaluate.load("bleu")
            bertscore = evaluate.load("bertscore")

            rouge_res = rouge.compute(predictions=predictions, references=references)
            bleu_res = bleu.compute(predictions=predictions, references=references)
            bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")

            # Accuracy, edit distance
            data["Exact_Match"] = data.apply(lambda x:
                x["Model_Output"].strip().lower() == x["Expected"].strip().lower(), axis=1)
            data["Edit_Distance"] = data.apply(lambda x:
                edit_distance(x["Model_Output"], x["Expected"]), axis=1)
            accuracy = data["Exact_Match"].mean() * 100

        st.success("✅ Evaluation Completed!")
        st.write("### 📋 Model Results Comparison")
        st.dataframe(data, use_container_width=True)

        # Executive Summary
        avg_edit = data["Edit_Distance"].mean()
        st.markdown(f"""
        ### 📄 Model Summary
        - **ROUGE-L:** {rouge_res['rougeL']:.4f}
        - **BLEU:** {bleu_res['bleu']:.4f}
        - **BERTScore (F1):** {np.mean(bert_res['f1']):.4f}
        - **Accuracy:** {accuracy:.2f}%
        - **Average Edit Distance:** {avg_edit:.2f}
        **Interpretation:**  
        High ROUGE/BERTScore → strong grammar & meaning.  
        BLEU < 1 due to natural paraphrasing.  
        Accuracy is low (AI rewrites style).  
        Edit Distance shows typical word-level change.
        """)

        # Metrics Comparison Bar Chart (Altair)
        chart_data = pd.DataFrame({
            "Metrics": ["ROUGE‑L", "BLEU", "BERTScore (F1)", "Accuracy (%)"],
            "Values": [rouge_res['rougeL'], bleu_res['bleu'], np.mean(bert_res['f1']), accuracy / 100]
        })
        bar_graph = alt.Chart(chart_data).mark_bar(color="#1f77b4").encode(
            x=alt.X("Metrics", sort=None),
            y="Values",
            tooltip=["Metrics", "Values"]
        ).properties(width=500, height=300)
        st.altair_chart(bar_graph, use_container_width=True)

        st.write("### 🔢 Edit Distance Per Sentence")
        st.line_chart(data[["Edit_Distance"]])

        st.write("### 🥧 Correct vs Incorrect Sentences")
        pie_data = pd.DataFrame({
            "Result": ["Correct", "Incorrect"],
            "Count": [data["Exact_Match"].sum(), len(data) - data["Exact_Match"].sum()]
        })
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=40).encode(
            theta="Count", color="Result", tooltip=["Result", "Count"]
        ).properties(width=300, height=300)
        st.altair_chart(pie_chart, use_container_width=True)

        st.download_button("💾 Download Evaluation Report", data.to_csv(index=False),
                           "evaluation_report.csv", "text/csv")

st.markdown("<hr><center>Built with ❤️ using Streamlit, Transformers & LoRA</center>", unsafe_allow_html=True)
