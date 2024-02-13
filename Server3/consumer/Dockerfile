FROM python:3.9-slim-buster

ARG channel
ARG bs_id

ENV chan $channel
ENV bid $bs_id

WORKDIR /consumer

COPY consumer.py ./
COPY server_config.py ./

RUN echo "sent IP: $dist_ip"
RUN echo "sent channel: $channel"

CMD python3 -u consumer.py ${chan} ${bid}
