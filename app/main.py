import gradio as gr
import os
import openai
from dotenv import load_dotenv, find_dotenv

from model_runner import model_runner
from inference import inference
from utils import cleanup

from default_template import TEMPLATE

from app_state import SingletonState
import logging

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY'] 

state = SingletonState()

logging.basicConfig(level=logging.DEBUG)
with gr.Blocks() as demo:
    placeholder_template = TEMPLATE.strip()
    
    model_name = gr.Textbox(label='Model', info='/code/models/phi-2 | /code/models/CodeLlama-13b-Instruct-hf')
    with gr.Row():
        quant = gr.Dropdown(['full', '8bit', '4bit'], label='Quant')
        device = gr.Dropdown(['auto', 'gpu'], label='Device')
    b1 = gr.Button('Load Model')
    # progress = gr.Textbox()
    llm_type = gr.Radio(choices=['open_llm', 'gpt_3', 'gpt_4'], value='open_llm')
    temperature = gr.Slider(minimum=0, maximum=1, value=0.25, label='Temperature')
    message = gr.Textbox(label='Message')
    template = gr.TextArea(value=None, label='Custom Template', placeholder=placeholder_template)
    bot = gr.Chatbot(label='Response')
    b2 = gr.Button('Submit')
    b3 = gr.Button('Unload Model')
    b1.click(model_runner, inputs=[model_name, quant, device])
    b2.click(inference, inputs=[message, bot, llm_type, temperature, template], outputs=[message, bot])
    b3.click(cleanup)
    
    

if __name__ == "__main__":   
    demo.launch(server_name="0.0.0.0", server_port=7860)
    