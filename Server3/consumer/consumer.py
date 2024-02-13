import os
import sys
import time
import socket
from threading import Thread
import multiprocessing
import numpy as np

sys.path.append('..')
from server_config import *
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa


# processData from incoming buffer, extract contents, demodulate and decode, then reply to client with decoding success
#
#   in:     file:    Raw active session samples to be compressed
#
#   out:    None
def process_data(file_from, respQ):
    # First extract file information
    file, addr = file_from

    pkt_num = np.frombuffer(file[8:12], dtype=np.int32)
    bs_fs = np.frombuffer(file[12:16], dtype=np.int32)
    samp_time = np.frombuffer(file[16:24], dtype=np.float64)
    IQ = np.frombuffer(file[24:], dtype=np.complex64)


    print(f"Working on AP: {pkt_num}")

    num_dec = [0, 0, 0, 0, 0, 0]
    response = [0, 0, pkt_num]
    # demoder = Std_LoRa(NUM_PREAMBLE, NUM_SYNC, NUM_DC, MAX_DATA_SYM, HAS_CRC)
    #
    # for SF in LORA_SF:
    #    pkts = demoder.Evaluate(IQ, SF, LORA_BW, bs_fs, True)
    #    num_dec[SF - 7] += len(pkts)
    #    for thing in pkts:
    #        # logging.append((pkt.channel, pkt.pkt_num, pkt.start_time, SF, thing[2]))
    #        response = response + thing[2].tolist()

    # normally just respond with num decoded
    tester = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1], dtype=np.float64)
    response = response + tester.tolist()
    response[0] = len(response) * 4 - 4
    response[1] = sum(num_dec)

    rewards = np.asarray(response, dtype=np.int32)
    respQ.put((addr, rewards))


# a process in a pool of processes that receives from a global queue to process data
#
#   in:     input_queue:    a multiprocessing queue filled with compressed files from clients
#           resp_queue:     a global queue that we place responses into after decoding is complete
#
#   out:    None
def worker(input_queue, resp_queue):
    while True:
        if not input_queue.empty():
            process_data(input_queue.get(), resp_queue)
        else:
            time.sleep(0.1)


# Continually send updates on packet decoding success rates
#   in:     None
#   out:    None
def process_responses():
    while True:
        if not responses.empty():
            resp = responses.get()
            if resp[0] in response_ports:
                resp_sock = response_ports[resp[0]]
                resp_sock.send(resp[1].tobytes())
        else:
            time.sleep(0.1)


responses = multiprocessing.Queue()     # queue of responses to be sent back
response_ports = {}
sessions = multiprocessing.Queue()      # queue of client packets

if __name__ == "__main__":
    # ch_id = sys.argv[1]
    # bs_id = sys.argv[2]

    rewarder = Thread(target=process_responses, args=[])
    rewarder.start()

    # spawn some processes to work on incoming data
    myPool = multiprocessing.Pool(2, worker, (sessions, responses))

    # attempt to connect to server (distributor)
    flow = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    channel = 1
    port = 63200 + channel
    server_ip = socket.gethostbyname(socket.gethostname())
    address = (server_ip, port)
    try:
        flow.connect(address)
        print("Connected to server")
        print(address)

        full_msg = b''
        new_msg = True
        msglen = 0
        num_rec = 0

        response_ports[address] = flow

        while True:
            try:
                full_msg += flow.recv(2 ** 20)
            except Exception as e:
                response_ports.pop(address)
                break

            if new_msg:
                msglen = int.from_bytes(full_msg[:4], 'big')
                new_msg = False

            if len(full_msg) >= msglen + 4:
                num_rec += 1
                sessions.put((full_msg, address))

                full_msg = full_msg[msglen + 4:]
                new_msg = True

    except Exception as e:
        print("Failed to connect to server")

