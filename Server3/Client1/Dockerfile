FROM python:3.9-slim-buster

ARG channel
ARG bs_id

ENV chan $channel
ENV bid $bs_id

WORKDIR /Client1

COPY Client1.py ./

EXPOSE 2004

RUN echo "sent IP: $dist_ip"
RUN echo "sent channel: $channel"

CMD python3 -u Client1.py ${chan} ${bid}
