from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from default_template import TEMPLATE
from app_state import SingletonState

def sanitize_str(inp):
    return inp.replace('{', '{{').replace('}', '}}')

def desanitize_str(inp):
    return inp.replace('{{', '{').replace('}}', '}')

def inference(message, history, llm_type, temperature, template=None):
    app_state = SingletonState()
    sanitised_message = sanitize_str(message)
    history_transformer_format = history + [[sanitised_message, ""]]
    history_str = "".join(["".join(["\n<question>"+item[0], "\n<answer>"+item[1]])  #curr_system_sanitised_message +
                for i, item in enumerate(history_transformer_format) if i > 0])
    result_template = template if template is not None and template != "" else TEMPLATE.strip()
    prompt = PromptTemplate(template=result_template.replace('{history}', sanitize_str(history_str)), input_variables=["input"])
    if (llm_type == 'gpt_3'):
        gpt3_llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo") # convert scale to 0 - 2 for GPT models.
        chain = LLMChain(llm=gpt3_llm, prompt=prompt, verbose=True)
        response = chain.invoke({"input": sanitised_message})
    elif (llm_type == 'gpt_4'):
        gpt4_llm = ChatOpenAI(temperature=temperature, model="gpt-4-0125-preview")
        chain = LLMChain(llm=gpt4_llm, prompt=prompt, verbose=True)
        response = chain.invoke({"input": sanitised_message})
    else:
        try:
            app_state.data["llm_obj"]["chain"] = LLMChain(llm=app_state.data["llm_obj"]["open_llm"], prompt=prompt, verbose=True)
            response = app_state.data["llm_obj"]["chain"].invoke({"input": sanitised_message})
        except KeyError:
            raise(Exception('Missing Open LLM model'))

    history.append((message, response["text"]))
    
    # response = chain.predict(input=text)
    return "", history 