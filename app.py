import os
import streamlit as st
from huggingface_hub import InferenceClient

# Page Config
st.set_page_config(
    page_title="Ayurvedic Herb-Drug Safety Checker",
    page_icon="🌿",
    layout="centered"
)

# Header & Disclaimer
st.title("🌿 Ayurvedic Herb-Drug Safety Checker")
st.caption("AI-powered clinical interaction analysis between Western pharmaceuticals and Ayurvedic herbs.")

st.warning(
    "⚠️ **Disclaimer:** This application is an experimental AI demonstration for educational "
    "and portfolio purposes only. It is not intended for medical diagnosis, triage, or treatment guidance. "
    "Always consult a qualified medical doctor and certified Ayurvedic physician."
)

st.divider()

# Input UI
st.subheader("Check Herb-Drug Compatibility")
col1, col2 = st.columns(2)

with col1:
    pharmaceutical = st.text_input("Modern Pharmaceutical / Medication", placeholder="e.g., Aspirin, Metformin")

with col2:
    ayurvedic_herb = st.text_input("Ayurvedic Herb / Formulation", placeholder="e.g., Guggulu, Gurmar")

# Retrieve Hugging Face API Token from Secrets
hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

# Inference Execution
if st.button("Analyze Safety Interaction", type="primary"):
    if not pharmaceutical or not ayurvedic_herb:
        st.error("Please enter both a modern medication and an Ayurvedic herb to analyze.")
    elif not hf_token:
        st.error("Hugging Face API token missing. Please configure HF_TOKEN in Streamlit secrets.")
    else:
        with st.spinner("Analyzing pharmacological and phytochemical interactions..."):
            try:
                # Initialize Hugging Face Inference Client
                client = InferenceClient(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    token=hf_token
                )

                system_prompt = (
                    "You are an AI Ayurvedic Herb-Drug Safety Checker. "
                    "Analyze potential contraindications between modern pharmaceuticals and Ayurvedic herbs. "
                    "Always provide clear safety ratings, potential biochemical risks, and modern precautions."
                )
                
                user_prompt = f"Check interaction between {pharmaceutical} and {ayurvedic_herb}."

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.2)
                output_text = response.choices[0].message.content

                st.success("Analysis Complete")
                st.markdown(output_text)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

st.divider()
st.caption("Built with Unsloth • Llama-3.2-3B • Streamlit")
