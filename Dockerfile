FROM python:3.9-slim-buster

WORKDIR /Server

COPY Server_Reqs.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY Server ./

EXPOSE 1234

CMD ["python3", "-u", "server.py"]