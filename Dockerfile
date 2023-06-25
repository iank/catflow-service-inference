FROM python:3.10 as base

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir .

EXPOSE 5057

CMD ["catflow-service-inference"]
