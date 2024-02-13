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
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa

# processData from incoming buffer, extract contents, demodulate and decode, then reply to client with decoding success
#
#   in:     file:    Raw active session samples to be compressed
#
#   out:    None
def process_data(file_from, respQ):
    # First extract file information
    file, addr = file_from
    msglen = int.from_bytes(file[:4], 'big')
    z = lz4.frame.decompress(file[12:msglen + 12])  # unzip file
    pkt = Active_Session.from_packet(z)

    print(f"{os.getpid()}, is working on packet: {pkt.pkt_num}")

    num_dec = [0, 0, 0, 0, 0, 0]

    logging = list()
    # (pkt.channel, pkt.pkt_num, pkt.start_time, SNR, decoded_Symbols)

    demoder = Std_LoRa(NUM_PREAMBLE, NUM_SYNC, NUM_DC, MAX_DATA_SYM, HAS_CRC)

    for SF in LORA_SF:
        pkts = demoder.Evaluate(pkt.buffer, SF, LORA_BW, FS, True)
        num_dec[SF - 7] += len(pkts)
        for thing in pkts:
            logging.append((pkt.channel, pkt.pkt_num, pkt.start_time, SF, thing[2]))

    rewards = [12, int(pkt.pkt_num), int(pkt.channel), int(sum(num_dec))]
    rewards = np.asarray(rewards, dtype=np.int32)
    respQ.put(((addr, rewards), logging))


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


# handle_client continually handle multiple clients connecting to server, make sure to receive full buffer to un-Lz4
#
#   in:     clientsocket:    A client socket attempting to send data
#           address:         Client's address
#
#   out:    None
def handle_client(clientsocket, address):
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


# Continually send updates on packet decoding success rates
#   in:     None
#   out:    None
def process_responses():
    logger = list()
    start_time = time.time()
    while True:
        if not responses.empty():
            resp, AP_log = responses.get()
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
        for b in a:
            f1.write(f"{int(b[0])},")  # log channel, active period, time, SF, decoded data
            f1.write(f"{int(b[1])},")
            f1.write(f"{b[2]},")
            f1.write(f"{int(b[3])},")
            f1.write(f"{b[4].tolist()}\n")
    f1.close()


responses = multiprocessing.Queue()     # queue of responses to be sent back
response_ports = {}
sessions = multiprocessing.Queue()      # queue of client packets

if __name__ == "__main__":
    rewarder = Thread(target=process_responses, args=[])
    rewarder.start()

    # spawn some processes to work on incoming data
    myPool = multiprocessing.Pool(NUM_CORES, worker, (sessions, responses))

    # server always listens for incoming connections
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((MY_IP, UPLINK_PORT))
    s.listen(8)

    while True:
        sock, address = s.accept()
        connect = Thread(target=handle_client, args=(sock, address))
        connect.start()



