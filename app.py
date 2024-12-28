import streamlit as st
import time
from chain_builder import rag_query  # Import your RAG chain from the main module

st.title("Medical Guideline Assistant")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role":
        "assistant",
        "content":
        "How can I help you with the information about medical guidelines"
    }]

# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input handling
if prompt := st.chat_input():
    # Append the user's query to the messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message
    st.chat_message("user").write(prompt)

    # Create a placeholder for the assistant's response
    assistant_message = st.chat_message("assistant").empty()

    # Create a generator function for streaming
    def stream_response(query, history):
        # Get response from RAG chain
        full_response = ""
        for chunk in rag_query(query, history, stream=True):
            full_response += chunk
            words = full_response.split()
            assistant_message.markdown(' '.join(words))
            time.sleep(0.05)  # Small delay between words
        return full_response

    # Stream the response
    response = stream_response(prompt, st.session_state.messages)

    # Update the final message in session state with the assistant's response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Rerun to refresh the entire chat history display
    st.rerun()