import numpy as np
import os
from scipy import signal
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa
import matplotlib.pyplot as plt

def channelize(master, shifter, filt):

    # shift to baseband
    chan = np.multiply(shifter, master)

    # finally downsample to 4x upsampling factor
    if UPSAMPLE_FACTOR == 16:
        chan = signal.resample_poly(chan, 10, 11)
    else:
        chan = signal.decimate(chan, int(RAW_FS / FS), ftype=filt, zero_phase=False)

    return chan.astype(np.complex64)


CENTER_FREQ = 903.3e6
RAW_FS = 2e6

FREQ_OFF = 200e3
FC = 70e3

UPSAMPLE_FACTOR = 4  # factor to downsample to
FS = 125000 * UPSAMPLE_FACTOR  # recerver's sampling rate
BW = 125000  # LoRa bandwidth

## build_filter
b, a = signal.ellip(4, 1, 100, FC / (RAW_FS / 2), 'low', analog=False)
myfilt = signal.dlti(b, a)

FILE_PATH = 'C:/Users/UWcon/Desktop/RL_Data/Outdoor_Test2/'

num_preamble = 8
num_sync = 2
num_DC = 2.25
num_data_sym = 50
upsampling_factor = 4

# load in plain buffer
# Channels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
#Channels = [-3, -2, 1, 3, 4]
Channels = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
overlap = 10 * RAW_FS / BW

direc = FILE_PATH
files = os.listdir(direc)

for file in files:
    fname = direc + file
    if file != '3_thur_11_resamp' and file != '3_thur_12_resamp' and file != '3_thur_13_resamp' \
            and file != '3_thur_14_resamp' and file != '3_thur_15_resamp':
        continue
    print(f"Processing File: {file}")
    file_size = os.path.getsize(fname)
    total_dec = 0

    for channel in Channels:

        phaser = 1j * 2 * np.pi * channel * (FREQ_OFF / RAW_FS)
        phaser = np.exp(np.arange(1, RAW_FS + (overlap * 2) + 1) * phaser)

        RX_input = np.ndarray((0,), dtype=np.complex64)
        offset = 0
        while offset + RAW_FS * 8 < file_size - 100:
        # while offset + RAW_FS * 8 < file_size // 8:
            f = open(fname)
            chunk = np.fromfile(f, dtype=np.complex64, count=int(RAW_FS + overlap * 2), offset=int(offset))
            f.close()

            chunk = channelize(chunk, phaser, myfilt)
            chunk = chunk[10 * UPSAMPLE_FACTOR: -10 * UPSAMPLE_FACTOR]

            RX_input = np.append(RX_input, chunk)
            offset += RAW_FS * 8

        SFs = [7, 8, 9]
        num_dec = 0
        for SF in SFs:
            # demodulate packet
            demoder = Std_LoRa(num_preamble, num_sync, num_DC, num_data_sym)
            num_pkts = demoder.Evaluate(RX_input, SF, BW, FS)
            num_dec += num_pkts

        print(f"{channel}\t{num_dec}\t{0}")
