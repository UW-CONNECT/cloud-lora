FROM python:3.9-slim-buster

ARG port_t

ENV my_port $port_t

WORKDIR /End_User

COPY consumer.py ./
COPY End_User_config.py ./
COPY Demodulators ./Demodulators

RUN pip3 install numpy==1.20.3
RUN pip3 install scipy==1.7.1

RUN ls

RUN echo "Listen to port: $port_t"

CMD python3 -u consumer.py ${my_port}
