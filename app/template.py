

import json
import logging
import gradio as gr
from default_template import TEMPLATE
from app_state import SingletonState


def load_tmpl():
    with open('templates/templates.json') as f:
        data = json.loads(f.read())
        state = SingletonState()
        state.data["templates"] = data["templates"]
        return gr.Radio(choices=list(state.data["templates"].keys()), visible=True, interactive=True), gr.Button(visible=True)

def load_selected(key):
    try:
        state = SingletonState()
        logging.debug(key)
        return state.data["templates"][key]
    except:
        logging.debug('errorrrr')
        return TEMPLATE.strip()