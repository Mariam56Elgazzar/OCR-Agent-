import streamlit as st
from typing import TypedDict, Optional, Annotated
import os
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import base64

# NOTE: The 'agent.tools' module must be available and contain the 'extract_text' function.
# If extract_text relies on an LLM for vision (which the description suggests), ensure 
# your Groq model supports multimodal or switch to a multimodal model like llama-4-scout.
from agent.tools import extract_text

# --- 1. CONFIGURATION AND INITIALIZATION ---

load_dotenv()

# Streamlit App Setup
st.set_page_config(page_title="LangGraph OCR Agent", layout="wide")
st.title("📄 LangGraph OCR Agent powered by Groq")
st.caption("Upload an image, and the agent will use the `extract_text` tool to transcribe it.")

# --- 2. LANGGRAPH/AGENT DEFINITIONS (Moved from main.py) ---

class AgentState(TypedDict):
    input_file: Optional[str] # Now a temporary file path
    messages: Annotated[list[AnyMessage], add_messages]

tools = [
    extract_text,
]
groq_api_key = os.environ.get("GROQ_API_KEY")

# IMPORTANT: Reverting to the Llama 3.1 8B model. If your 'extract_text' tool
# *requires* the Groq model itself to handle vision (multimodal), you MUST 
# use a model like "meta-llama/llama-4-scout-17b-16e-instruct"
# If 'extract_text' handles the vision/base64 encoding internally, 'llama-3.1-8b-instant' is fine.
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    groq_api_key=groq_api_key
)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

def assistant(state: AgentState):
    # This system message is crucial. We must explicitly instruct the LLM to 
    # output the transcribed content, not just a disclaimer.
    textual_description_of_tools = """
    def extract_text(img_path: str) -> str:
        # ... (description as before) ...
    """
    
    image_path = state["input_file"]
    sys_msg = SystemMessage(content = f"""
    You are a helpful assistant specialized in Optical Character Recognition (OCR).
    You MUST use the `extract_text` tool on the provided image path: {image_path}.
    After the tool returns the text, your final response must be the **FULL transcribed text**
    from the image, clearly formatted under a 'Transcription Result' heading.
    You can analyse documents with provided tools:\n{textual_description_of_tools}.
    """)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }

# LangGraph Build
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# --- 3. STREAMLIT UI AND EXECUTION LOGIC ---

uploaded_file = st.file_uploader("Upload an Image for Transcription", type=["png", "jpg", "jpeg"])

user_prompt = "Please transcribe the provided image."

if uploaded_file is not None:
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    with col2:
        st.subheader("Agent Output")
        
        # 1. Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            # 2. Setup initial state and run the graph
            initial_messages = [HumanMessage(content=user_prompt)]
            
            # Use st.spinner to show progress
            with st.spinner(f"Running LangGraph agent on {uploaded_file.name} using {llm.model_name}..."):
                
                final_state = react_graph.invoke({
                    "messages": initial_messages,
                    "input_file": temp_file_path
                })
                
            # 3. Extract and display the final result
            final_message = final_state['messages'][-1]
            
            if final_message.content:
                st.success("Transcription Complete!")
                st.markdown(final_message.content)
            else:
                # If the final message is empty, we look for the ToolMessage content
                tool_output_message = next((m for m in final_state['messages'] if isinstance(m, ToolMessage)), None)
                if tool_output_message:
                    st.warning("Agent's final response was empty. Displaying direct tool output:")
                    st.code(tool_output_message.content, language="text")
                else:
                    st.error("Agent failed to produce a transcription or a final message.")

        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            st.code(final_state) # Optional: print the last state for debugging
            
        finally:
            # 4. Clean up the temporary file
            os.unlink(temp_file_path)

else:
    st.info("Please upload an image to begin OCR transcription.")