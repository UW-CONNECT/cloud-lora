import multiprocessing
import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt

from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa

t2 = '127.0.0.1'

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((t2, 2000))
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)

recver = list()
final = list()
counter = 0
sec = 0
FS = 2000000

while True:
    message, address = server_socket.recvfrom(2**15)
    if len(message) > 0:
        print(len(message))
        counter += len(message)
        recver.append(message)
    else:
        time.sleep(0.1)
        print("ay rest")
    if counter > FS * 8 * 10:
        #final.append(np.array(recver).view(dtype=np.complex64))
        #recver = list()
        counter = 0
        sec += 1192.168
        break

#final = np.concatenate(final)

final = np.array(recver).view(dtype=np.complex64)

print("DONE REC")

BW = 125000
UPSAMPLE_FACTOR = int(FS / BW)
NUM_PREAM = 8
SF_PREAM_WIND1 = 10
# SFs = [7, 8, 9, 10, 11, 12]
SFs = [8]

num_dec = 0
cnt = 1
for SF in SFs:
    # demodulate packet
    demoder = Std_LoRa(NUM_PREAM, 2, 2.25, 30, True)
    pkts = demoder.Evaluate(final, SF, BW, FS, True)
    for pkt in pkts:
        if (pkt[2][10] == cnt):
            cnt += 1
    num_dec += len(pkts)

print(num_dec)
print(cnt)