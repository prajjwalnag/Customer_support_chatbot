Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Posts
Docs
Enterprise
Pricing



Spaces:

Prajjwalng
/
customercare


like
0

App
Files
Community
Settings
customercare
/
app.py

Prajjwalng's picture
Prajjwalng
Update app.py
e3cc7e5
verified
less than a minute ago
raw

Copy download link
history
blame
edit
delete

4.02 kB
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from huggingface_hub import login
from peft import PeftModel, PeftConfig
import time

# Login with HF_TOKEN (if available)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        st.success("Hugging Face login successful!")
    except Exception as e:
        st.error(f"Hugging Face login failed: {e}")
else:
    st.warning("HF_TOKEN environment variable not set. Some features may be limited.")

# Model and Adapter Configuration
model_id = "Prajjwalng/gemma_customer_care"  # Base model
adapter_id = "Prajjwalng/gemma_customercare_adapters"  # adapter model

# Initialize model and tokenizer (load only once)
@st.cache_resource
def load_model(model_id):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    return base_model, tokenizer

merged_model, tokenizer = load_model(model_id)

# Function to generate chatbot response using the provided template
def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    prompt_template = f"""
<start_of_turn>system You are a support chatbot who helps with user queries chatbot who always responds in the style of a professional.\n<end_of_turn>
<start_of_turn>user
{query}
<end_of_turn>
<start_of_turn>model
"""
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    model.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    model_response = decoded.split("model\n")[-1].strip()
    return model_response

# Streamlit app
st.title("Customer Care ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial welcome message
    initial_message = {"role": "assistant", "content": "Hi, I am Sora, I am your customer support agent."}
    st.session_state.messages.append(initial_message)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display chatbot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        typing_placeholder = st.empty()
        typing_dots = ""  # Initialize empty string for typing dots

        # Animate typing dots
        for i in range(3):
            typing_dots += "."
            typing_placeholder.markdown(typing_dots)
            time.sleep(0.3)  # Adjust speed as needed

        typing_placeholder.empty()  # Clear typing dots

        full_response = ""
        response = get_completion(prompt, merged_model, tokenizer)

        # Simulate stream of responses with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a placeholder to stream the response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})