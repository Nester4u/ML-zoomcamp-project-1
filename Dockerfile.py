FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "Dockerfile.py", "./"]

EXPOSE 5000

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:5000", "predict:app" ]