from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from default_template import TEMPLATE
from app_state import SingletonState

def inference(message, history, llm_type, temperature, template=None):
    app_state = SingletonState()
    history_transformer_format = history + [[message, ""]]
    history_str = "".join(["".join(["\n<question>"+item[0], "\n<answer>"+item[1]])  #curr_system_message +
                for item in history_transformer_format])
    result_template = template if template is not None and template != "" else TEMPLATE.strip()
    prompt = PromptTemplate(template=result_template.replace('{history}', history_str), input_variables=["input"])
    if (llm_type == 'gpt_3'):
        gpt3_llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
        chain = LLMChain(llm=gpt3_llm, prompt=prompt, verbose=True)
        response = chain.invoke({"input": message})
    elif (llm_type == 'gpt_4'):
        gpt4_llm = ChatOpenAI(temperature=0.5, model="gpt-4-0125-preview")
        chain = LLMChain(llm=gpt4_llm, prompt=prompt, verbose=True)
        response = chain.invoke({"input": message})
    else:
        app_state.data["llm_obj"]["chain"] = LLMChain(llm=app_state.data["llm_obj"]["open_llm"], prompt=prompt, verbose=True)
        response = app_state.data["llm_obj"]["chain"].invoke({"input": message})

    history.append((message, str(response["text"])))
    
    # response = chain.predict(input=text)
    return "", history  