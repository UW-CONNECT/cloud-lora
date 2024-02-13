from scipy import signal
import numpy as np
from Demodulators.Standard_LoRa.Std_LoRa import Std_LoRa
import os


def channelize(master, shifter):
    ## build_filter
    b, a = signal.ellip(4, 1, 100, 70e3 / (2.2e6 / 2), 'low', analog=False)

    # shift to baseband
    chan = np.multiply(shifter, master)

    # then filter the channel to 125kHz
    chan = signal.lfilter(b, a, chan)

    # finally downsample to 4x upsampling factor
    if UPSAMPLE_FACTOR == 16:
        chan = signal.resample_poly(chan, 10, 11)
    else:
        chan = signal.resample_poly(chan, 5, int(88 / UPSAMPLE_FACTOR))

    return chan.astype(np.complex64)


def remove_sim(vals, dist):
    if len(vals) <= 1:
        return vals
    ret = []
    ret.append(vals[0])
    for i, _ in enumerate(vals):
        if i == 0:
            continue
        else:
            if vals[i] - vals[i-1] > dist:
                ret.append(vals[i])
    return ret

# store formatted data to a file location
def store_results(fname, value):
    f1 = open(fname, 'a')
    for a in value:
        tmp = a.tolist()
        f1.write(f"{tmp[7:11]}\n")  # log channel, active period, time, SF, decoded data
    f1.close()


if __name__ == "__main__":

    # Filter
    fc = 70e3
    fs = 500e3

    channels = [0]

    # params
    BW = 125000
    FS = 500000
    UPSAMPLE_FACTOR = int(FS / BW)

    SFs = [7, 8, 9]
    num_preamble = 8
    num_sync = 2
    num_DC = 2.25
    num_data_sym = 100
    freqoff = 200e3

    dir = 'C:/Users/danko/Desktop/RL_Data/Outdoor_test/'
    files = os.listdir(dir)
    fname = dir + '1chan_Dan_test'
    f = open(fname)

    buff = np.fromfile(f, dtype=np.complex64)
    f.close()

    AP_pkts = 0
    logger = list()

    for SF in SFs:
        demoder = Std_LoRa(num_preamble, num_sync, num_DC, num_data_sym, True)
        num_pkts = demoder.Evaluate(buff, SF, BW, FS, True)
        AP_pkts += len(num_pkts)
        for a in num_pkts:
            logger.append(a[2])

    store_results('log2.txt', logger)

