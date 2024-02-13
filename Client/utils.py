from scipy import signal
import numpy as np
import time
import DC_gen
#from Client.client_config import *
from client_config import *
from numba import jit
import os


# store formatted data to a file location
def store_results(fname, value):
    f1 = open(fname, 'a')
    f1.write(f"{value}\n")
    f1.close()


# function to detect the bandwidth estimates provided by BBR, if can't detect, just return large BW, that state
# input will act as a dummy variable and the RL will have to infer BW through latency and pkt sizes
def BW_sim():
    if OS_CONF == 'Linux':
        try:
            stream = os.popen('ss -tin | grep -A 1 "' + SERVER_IP + ':65432"')
            mystr = stream.read()
            idx = mystr.index('bw:')
            mystr = mystr[idx + 3:]
            idx = mystr.index('bps')
            mystr = int(mystr[:idx])
            if mystr == 0:
                return BACKUP_BW
            return mystr
        except ValueError as e:
            return BACKUP_BW
    else:
        return BACKUP_BW

# function to get network delays to see latency characteristics
def RTT_sim():
    if OS_CONF == 'Linux':
        try:
            stream = os.popen('ss -tin | grep -A 1 "' + SERVER_IP + ':65432"')
            mystr = stream.read()
            idx = mystr.index('rtt:')
            mystr = mystr[idx + 4:]
            idx = mystr.index('/')
            mystr = float(mystr[:idx])
            if mystr == 0:
                return 50
            return mystr / 2000
        except ValueError as e:
            return 50
    else:
        return 50


# Function to filter the bandwidth queue based on time
def trim_by_time(curr_time, q, t):
    while len(q) > 0 and curr_time - q[0][0] >= t:
        q.pop(0)


# function to get the overall active period sending-rate of a given channel
def get_rate(Prev_Send, curr_time):
    cumulative_sizes = 0
    if len(Prev_Send) == 0:
        return 1
    for index in Prev_Send:
        cumulative_sizes += index[1]
    time_diff = curr_time - Prev_Send[-1][0]
    if time_diff == 0:
        return cumulative_sizes
    return cumulative_sizes / time_diff


# function to integrate the bandwidth over the last n active periods, if -1, all active periods
def integrate_BW(Prev_Send, curr_time, n):
    integral = 0
    cumulative_sizes = 0
    if len(Prev_Send) == 0:
        return 1, 1
    if n == -1:
        if len(Prev_Send) > 1:
            for i in range(len(Prev_Send) - 1):
                time_interval = Prev_Send[i + 1][0] - Prev_Send[i][0]
                integral += (time_interval * Prev_Send[i][2])
                cumulative_sizes += Prev_Send[i][1]
        integral += (curr_time - Prev_Send[-1][0]) * Prev_Send[-1][2]
        cumulative_sizes += Prev_Send[-1][1]
    else:
        for i in range(min(len(Prev_Send), n) - 1):
            time_interval = Prev_Send[-i - 1][0] - Prev_Send[-i - 2][0]
            integral += (time_interval * Prev_Send[-i - 2][2])
            cumulative_sizes += Prev_Send[-i - 2][1]
        integral += (curr_time - Prev_Send[-1][0]) * Prev_Send[-1][2]
        cumulative_sizes += Prev_Send[-1][1]
    if integral == 0:
        integral = 1
    if cumulative_sizes == 0:
        cumulative_sizes = 1
    return integral, cumulative_sizes


# helper function to get the previous success rates of sent active periods. This normalizes the rewards per AP to +1
def get_prev_success(Prev_Send):
    ratio = 1
    if len(Prev_Send) > 0:
        ratio = sum(Prev_Send) / len(Prev_Send)

    if ratio == 0:
        ratio = 1

    return ratio


# In the case of fragmentation, this helper function will perform a weighted combination of each active period
#   in:     sess1&2 are active periods to combine
#   out:    combined statistics on each active period
def weighted_sess_combine(sess1, sess2):
    len1 = sess1[1] - sess1[0]
    len2 = sess2[1] - sess2[0]
    weight1 = len1 / (len1 + len2)
    weight2 = len2 / (len1 + len2)
    sess1[1] += len2
    if sess2[2] == 2:
        sess1[2] = 0
    else:
        sess1[2] = 1
    sess1[3] = sess1[3] * weight1 + sess2[3] * weight2
    for i, bucket in enumerate(sess1[4]):
        sess1[4][i] = bucket * weight1 + sess2[4][i] * weight2
    sess1[5] = sess1[5] * weight1 + sess2[5] * weight2

    return sess1


# Helper function to convert a channel to baseband and filter
#   in:     master is raw input samples
#           channel is the channel we wish to decode
#   out:    a baseband, filtered and downsampled stream
def channelize(master, shifter, filt):
    # shift to baseband
    chan = np.multiply(shifter, master)

    # finally downsample to 4x upsampling factor
    if UPSAMPLE_FACTOR == 16:
        chan = signal.resample_poly(chan, 10, 11)
    else:
        chan = signal.decimate(chan, int(RAW_FS / FS), ftype=filt, zero_phase=False)

    return chan.astype(np.complex64)


def build_super_DC(spread_factor):
    DCL = DC_gen.DC_gen(spread_factor, BW, FS)

    DCM = DC_gen.DC_gen(spread_factor - 1, BW, FS)
    DCM = np.tile(DCM, 2)

    DCS = DC_gen.DC_gen(spread_factor - 2, BW, FS)
    DCS = np.tile(DCS, 4)
    Super_DC = DCL + DCM + DCS

    return Super_DC


@jit(nopython=True)
def hash_set(gains):
    gains2 = [(e - 2) * 4 for e in gains]
    hist = np.zeros(64, dtype=np.float64)
    for e in gains2:
        if (e < 0):
            hist[0] += 1
        elif (e > 63):
            hist[63] += 1
        else:
            hist[int(e)] += 1

    hist = hist / np.sum(hist)
    return hist


@jit(nopython=True)
def mini_hash_set(gains):
    gains2 = [(e - 2) / 4 for e in gains]
    hist = np.zeros(4, dtype=np.float64)
    for e in gains2:
        if (e < 0):
            hist[0] += 1
        elif (e > 3):
            hist[3] += 1
        else:
            hist[int(e)] += 1

    hist = hist / np.sum(hist)
    return hist


@jit(nopython=True)
def set_intersect(AS_12, AS_9):
    ret = []
    comb = AS_12 + AS_9
    comb.sort(key=lambda x: x[0])
    for interval in comb:
        if ret and interval[0] <= ret[-1][1]:  # connect section
            ret[-1][1] = max(interval[1], ret[-1][1])
            ret[-1][4] = min(interval[4], ret[-1][4])
        else:
            ret.append(interval)

    return ret

