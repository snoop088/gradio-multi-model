import gradio as gr
import os
import openai
from dotenv import load_dotenv, find_dotenv
from template import load_selected, load_tmpl

from model_runner import model_runner
from inference import inference
from utils import cleanup

from default_template import TEMPLATE

from app_state import SingletonState
import logging
import json

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY'] 

state = SingletonState()

logging.basicConfig(level=logging.DEBUG)



with gr.Blocks() as demo:
    
    # def load_tmpl():
    #     with open('templates/templates.json') as f:
    #         data = json.loads(f.read())
    #         state = SingletonState()
    #         state.data["templates"] = data["templates"]
    #         return {tmpls: gr.Radio(choices=list(state.data["templates"].keys()), visible=True, interactive=True), loader: gr.Button(visible=True)}

    # def load_selected(key):
    #     try:
    #         state = SingletonState()
    #         logging.debug(key)
    #         return state.data["templates"][key]
    #     except:
    #         logging.debug('errorrrr')
    #         return TEMPLATE.strip()
    
    placeholder_template = TEMPLATE.strip()
    
    model_name = gr.Textbox(label='Model', info='/code/models/phi-2 | /code/models/CodeLlama-13b-Instruct-hf')
    with gr.Row():
        quant = gr.Dropdown(['full', '8bit', '4bit'], label='Quant')
        device = gr.Dropdown(['auto', 'gpu'], label='Device')
        temperature = gr.Slider(minimum=0, maximum=1, value=0.25, label='Temperature')
    b1 = gr.Button('Load Model')
    # progress = gr.Textbox()
    llm_type = gr.Radio(choices=['open_llm', 'gpt_3', 'gpt_4'], value='open_llm')
    
    message = gr.Textbox(label='Message')
    template = gr.TextArea(value=None, label='Custom Template', placeholder=placeholder_template)
    b_tmpl_load = gr.Button('Load Templates')
    tmpls = gr.Radio(visible=False)
    loader = gr.Button('Load Selected', visible=False)
        
    bot = gr.Chatbot(label='Response', show_copy_button=True)
    b2 = gr.Button('Submit')
    b3 = gr.Button('Unload Model')
    b4 = gr.ClearButton([model_name, quant, device, message, template, bot] )
    b1.click(model_runner, inputs=[model_name, quant, device, temperature])
    b2.click(inference, inputs=[message, bot, llm_type, temperature, template], outputs=[message, bot])
    b3.click(cleanup)

    

    b_tmpl_load.click(load_tmpl, outputs=[tmpls, loader])
    loader.click(load_selected, inputs=[tmpls], outputs=[template])
    
    
        

if __name__ == "__main__":   
    demo.launch(server_name="0.0.0.0", server_port=7860)
    