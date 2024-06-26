FROM python:3.12

WORKDIR /code

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

WORKDIR /code/app

CMD [ "gradio", "main.py"]