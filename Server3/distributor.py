import socket
import os
import time
from threading import Thread
import multiprocessing
import numpy as np

# (id, chan):(IP, port)
port_map = {
    ('345', 1): ('192.168.1.9', 63211, multiprocessing.Queue()),
    ('678', 2): ('127.0.1.1', 63212, multiprocessing.Queue())}


def client_connection(clientsocket, address):
    print(f"Connection from {address} has been established, read input buffer")
    full_msg = b''
    new_msg = True
    msglen = 0
    num_rec = 0

    response_ports[address] = clientsocket

    while True:
        try:
            full_msg += clientsocket.recv(2 ** 20)
        except Exception as e:
            response_ports.pop(address)
            return
        if new_msg:
            msglen = int.from_bytes(full_msg[:4], 'big')
            new_msg = False

        if len(full_msg) >= msglen + 4:
            print(num_rec)
            num_rec += 1
            sessions.put((full_msg, address))

            full_msg = full_msg[msglen + 4:]
            new_msg = True


def consumer_connection(connection):
    flow = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        flow.connect((connection[0], connection[1]))
        print("Connected to server")
    except Exception as e:
        print("Failed to connect to server")

    # todo: just have a while loop continually looking to pull from queue and send data

    r_id = 345
    chan = 1
    fs = 500000
    pkt_time = time.time()

    header1 = np.asarray([r_id, chan, fs], dtype=np.int32)

    header2 = np.asarray([pkt_time], dtype=np.float64)

    data = np.random.uniform(-1, 1, 1010) + 1.j * np.random.uniform(-1, 1, 1010)
    data = data.astype(np.complex64)

    ################ build formatted file by concatting bytes then lz4 #####################
    fullpkt = header1.tobytes() + header2.tobytes() + data.tobytes()
    pkt = (len(fullpkt)).to_bytes(4, 'big') + fullpkt
    try:
        flow.send(bytes(pkt))
    except Exception as e:
        print("Failed to send to server, either lost connection, or BW constrained, try again")


response_ports = {}

if __name__ == "__main__":
    # set up connection with all consumers
    for consumer in port_map:
        data = port_map[consumer]
        print(data)
        multiprocessing.Process(target=consumer_connection, args=(data,)).start()

    # todo: Also need to spawn worker threads that decompress data (only need like 3 or 4)
    # todo: like previous server they just read from a queue and place into another queue

    # server always listens for incoming connections
    MY_IP = '127.0.0.1'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((MY_IP, 65432))
    s.listen(3)

    while True:
        sock, address = s.accept()
        connect = Thread(target=client_connection, args=(sock, address))
        connect.start()
