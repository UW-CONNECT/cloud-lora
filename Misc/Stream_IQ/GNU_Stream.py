import multiprocessing
import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt

from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa

t1 = '192.168.176.1'
t2 = '127.0.0.1'

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((t1, 2000))
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**25)
server_socket.listen(7)
conn, addr = server_socket.accept()

recver = list()
final = list()
counter = 0
sec = 0

print("made it")

while True:
    message = conn.recv(2**20)
    if len(message) > 0:
        print(len(message))
        counter += len(message)
        recver.append(message)
    else:
        time.sleep(0.005)
    if counter > 250000 * 8 * 30:
        final.append(np.array(recver).view(dtype=np.complex64))
        recver = list()
        counter = 0
        sec += 1
        break
    if sec >= 30:
        break

final = np.concatenate(final)

demoder = Std_LoRa(8, 2, 2.25, 30)

num_pkts = demoder.Evaluate(final, 8, 125000, 250000)

plt.plot(final.real)
plt.show()