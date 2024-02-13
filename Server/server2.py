import os
import sys
import time

import lz4.frame
import numpy as np
from threading import Thread
import multiprocessing

sys.path.append('..')
from server_config import *
from Active_Session import Active_Session


def consumer_connection_reverse(sock_t, addr, channel, respQ):
    full_msg = b''
    new_msg = True
    msglen = 0
    num_rec = 0

    while True:
        full_msg += sock_t.recv(16)

        if new_msg:
            msglen = np.frombuffer(full_msg[0:4], dtype=np.int32)
            new_msg = False

        if len(full_msg) >= msglen + 4:
            num_rec += 1

            msg = np.frombuffer(full_msg, dtype=np.int32)
            num_dec = msg[1]
            pkt_index = msg[2]

            # place data into a queue to send and log (like in server)
            rewards = [12, int(pkt_index), int(channel), int(num_dec)]
            rewards = np.asarray(rewards, dtype=np.int32)
            respQ.put(((addr, rewards), msg[:int(msglen + 4)]))

            full_msg = full_msg[int(msglen) + 4:]
            new_msg = True


def consumer_connection(connection, resp_q):
    # listen implementation
    listen_on_port = connection[0]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((MY_IP, listen_on_port))
    s.listen(1)

    while True:
        sock, address = s.accept()
        print(f"Consumer connected on port {listen_on_port}")
        break

    channel = listen_on_port - 63200
    connect = Thread(target=consumer_connection_reverse, args=(sock, connection[0], channel, resp_q))
    connect.start()

    while True:
        if not connection[1].empty():
            # get and decompress active period
            file = connection[1].get()
            msglen = int.from_bytes(file[:4], 'big')
            z = lz4.frame.decompress(file[12:])  # unzip file
            pkt = Active_Session.from_packet(z)

            # build generic IQ stream to send to any user
            rad_id = np.asarray([int(pkt.radio_ID)], dtype=np.uint32)
            header1 = np.asarray([pkt.pkt_num, pkt.Fs], dtype=np.int32)
            header2 = np.asarray([pkt.start_time], dtype=np.float64)
            data = pkt.buffer
            data = data.astype(np.complex64)
            print(pkt.pkt_num)

            ################ build formatted file by concatting bytes then lz4 #####################
            fullpkt = rad_id.tobytes() + header1.tobytes() + header2.tobytes() + data.tobytes()
            pkt = (len(fullpkt)).to_bytes(4, 'big') + fullpkt
            try:
                sock.send(bytes(pkt))
            except Exception as e:
                print("Failed to send to server, either lost connection, or BW constrained, try again")
        else:
            time.sleep(0.1)


# handle_client continually handle multiple clients connecting to server, make sure to receive full buffer to un-Lz4
#
#   in:     clientsocket:    A client socket attempting to send data
#           address:         Client's address
#
#   out:    None
def handle_client(clientsocket, address):
    print(f"Connection from Access Point: {address} has been established")
    full_msg = b''
    new_msg = True
    msglen = 0
    num_rec = 0

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
            msglen = int.from_bytes(full_msg[:4], 'big')
            bs_id = int.from_bytes(full_msg[4:8], 'big')
            bs_ch = int.from_bytes(full_msg[8:12], 'big')
            mapping = PORT_MAP[(bs_id, bs_ch)]
            mapping[1].put(full_msg)    # place in correct queue
            if mapping[0] not in response_ports:
                response_ports[mapping[0]] = clientsocket
            num_rec += 1
            full_msg = full_msg[msglen + 4:]
            new_msg = True


# Continually send updates on packet decoding success rates
#   in:     None
#   out:    None
def process_responses():
    logger = list()
    start_time = time.time()
    while True:
        if not responses.empty():
            resp, AP_log = responses.get()
            #print(resp[0])
            if resp[0] in response_ports:
                resp_sock = response_ports[resp[0]]
                resp_sock.send(resp[1].tobytes())
            if len(AP_log) > 0:
                logger.append(AP_log)
        else:
            time.sleep(0.1)
            if time.time() - start_time > 10:
                store_results(LOGGER_LOC, logger)
                logger = list()
                start_time = time.time()


# store formatted data to a file location
def store_results(fname, value):
    f1 = open(fname, 'a')
    for a in value:
        f1.write(str(a.tolist()) + "\n")
        # for b in a:
        #     f1.write(f"{int(b[0])},")  # log channel, active period, time, SF, decoded data
        #     f1.write(f"{int(b[1])},")
        #     f1.write(f"{b[2]},")
        #     f1.write(f"{int(b[3])},")
        #     f1.write(f"{b[4].tolist()}\n")
    f1.close()


# need to map consumer port to a sock
response_ports = {}
responses = multiprocessing.Queue()     # queue of responses to be sent back

if __name__ == "__main__":
    # set up connection with all consumers
    for consumer in PORT_MAP:
        data = PORT_MAP[consumer]
        multiprocessing.Process(target=consumer_connection, args=(data, responses)).start()

    rewarder = Thread(target=process_responses, args=[])
    rewarder.start()

    # server always listens for incoming connections
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((MY_IP, UPLINK_PORT))
    s.listen(3)

    while True:
        sock, address = s.accept()
        connect = Thread(target=handle_client, args=(sock, address))
        connect.start()



