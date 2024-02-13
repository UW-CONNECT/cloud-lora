import time

import matplotlib.pyplot as plt

from Client.Active_Period_Detector import Active_Period_Detector2
from utils import channelize
from scipy import signal
import numpy as np
from multiprocessing import Queue
from Client.client_config import IQ_SOURCE
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa

# store formatted data to a file location
def store_results(fname, value):
    f1 = open(fname, 'a')
    for a in value:
        tmp = a.tolist()
        f1.write(f"{tmp[7:11]}\n")  # log channel, active period, time, SF, decoded data
    f1.close()

# Filter
fc = 70e3
fs = 500e3

# params
BW = 125000
FS = 500000
UPSAMPLE_FACTOR = int(FS / BW)

SFs = [7, 8, 9]
num_preamble = 8
num_sync = 2
num_DC = 2.25
num_data_sym = 50
freqoff = 200e3
channel = -1

# overlap = 10 * fs / BW
# b, a = signal.ellip(4, 1, 100, fc / (fs / 2), 'low', analog=False)
# phaser = 1j * 2 * np.pi * channel * (freqoff / fs)
# phaser = np.exp(np.arange(1, fs + (overlap * 2) + 1) * phaser)

inq = Queue()
outq = Queue()
fin = list()
fin.append(outq)
# myAPD = Active_Period_Detector2(channel, inq, outq)
# myAPD.start_consuming()


# dir = 'C:/Users/danko/Desktop/RL_Data/Outdoor_test/'
# fname = dir + '1chan_Dan_test'
# file_size = os.path.getsize(fname)
# f = open(fname)
# Rx_input = np.fromfile(f, dtype=np.complex64)
# f.close()
#
# inq.put((Rx_input[0:FS], 0))
#
# inq.put((Rx_input, 1))
# time.sleep(45)

########################### start decoding APs ############################
# logger = list()
# AP_pkts = 0
# while not outq.empty():
#     buff = outq.get()[1]
#     stat = [len(buff)]
#
#     for SF in SFs:
#         demoder = Std_LoRa(num_preamble, num_sync, num_DC, num_data_sym, True)
#         num_pkts = demoder.Evaluate(buff, SF, BW, FS, True)
#         AP_pkts += len(num_pkts)
#         for a in num_pkts:
#             logger.append(a[2])
#
# store_results('log2.txt', logger)

IQ_SOURCE(fin)

overlap = 10 * FS / BW
phaser = 1j * 2 * np.pi * channel * (200e3 / FS)
phaser = np.exp(np.arange(1, FS + (overlap * 2) + 1) * phaser)
b, a = signal.ellip(4, 1, 100, 70e3 / (FS / 2), 'low', analog=False)
filt = signal.dlti(b, a)

Rx_buff = list()

while not fin[0].empty():
    startt = time.time()
    buff = fin[0].get()
    very_start = buff[1]
    buff = channelize(buff[0], phaser, filt)
    buff = buff[10 * UPSAMPLE_FACTOR:-10 * UPSAMPLE_FACTOR]
    Rx_buff.append(buff)

Rx_buff = np.concatenate(Rx_buff)

logger = list()
AP_pkts = 0

for SF in SFs:
    demoder = Std_LoRa(num_preamble, num_sync, num_DC, num_data_sym, True)
    num_pkts = demoder.Evaluate(Rx_buff, SF, BW, FS, True)
    AP_pkts += len(num_pkts)
    for a in num_pkts:
        logger.append(a[2])

store_results('log1.txt', logger)

###################################### test apd now ############################
# myAPD = Active_Period_Detector2(channel, inq, outq)
# myAPD.configNF(Rx_buff[0:FS], 'low')
#
# AS = myAPD.Active_Sess_Detect2(Rx_buff, 'low', 0)
#
# logger = list()
# AP_pkts = 0
# ap_num = 0
#
# for sess in AS:
#     tmp = Rx_buff[sess[0]:sess[1]]
#     print(ap_num)
#
#     # if ap_num >= 225:
#     #     plt.plot(tmp.real)
#     #     plt.show()
#
#     for SF in SFs:
#         demoder = Std_LoRa(num_preamble, num_sync, num_DC, num_data_sym, True)
#         num_pkts = demoder.Evaluate(tmp, SF, BW, FS, True)
#         AP_pkts += len(num_pkts)
#         for a in num_pkts:
#             logger.append(a[2])
#     ap_num += 1
#
# store_results('log2.txt', logger)

pass
