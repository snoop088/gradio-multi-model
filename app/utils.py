import torch
import gc
import logging
from app_state import SingletonState

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def cleanup():
    app_state = SingletonState()
    
    try:
        for key in app_state.data["llm_obj"].keys():
            app_state.data["llm_obj"][key] = None
        del app_state.data["llm_obj"] 
        app_state.rem_data()
        gc.collect()
        logging.debug(gc.garbage)
    except:
        logging.error(f'Error cleaning state. State is {app_state.data}')
    logging.debug('clearing cache')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
