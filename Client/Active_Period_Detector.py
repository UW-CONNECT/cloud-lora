import time
import matplotlib.pyplot as plt
import DWT_compress as DWT
from threading import Thread
from utils import *
import numpy as np
import math

"""
Active_Period_Detector pipeline:
[consumer]: A constantly running process (after configuration) that will maintain input buffer reconstruction, 
            active period fragmentation, and other necessary functions for active period detection

in_queue[] -> consumer -> out_queue[]
"""


class Active_Period_Detector2:

    def __init__(self, channel, in_queue, out_queue):
        # interface queues
        self.in_q = in_queue
        self.out_q = out_queue

        # active period detection constants
        self.win_jump_factor = 3
        self.num_keep = 3
        self.front_buf = 2 * self.win_jump_factor
        self.rear_buf = 4 * self.win_jump_factor

        # active period detection constants after initialization
        self.ASD_small_thresh_high = 0
        self.ASD_small_thresh_low = 0
        self.ASD_small_superDC = build_super_DC(9)
        self.ASD_big_thresh_high = 0
        self.ASD_big_thresh_low = 0
        self.ASD_big_superDC = build_super_DC(12)
        self.Noise_Avg = 0

        # internal state values
        self.curr_state = 0
        self.low_buffer = 0
        self.frag_store = None
        self.frag_buff = np.ndarray((0,), dtype=np.complex64)

        self.last_buff = np.ndarray((0,), dtype=np.complex64)

        np.random.seed(1234)

        # internal data to save
        print(f"Hey I am channel: {channel}")
        self.channel = channel
        self.center_freq = CENTER_FREQ + channel * FREQ_OFF
        overlap = 10 * RAW_FS / BW
        phaser = 1j * 2 * np.pi * self.channel * (FREQ_OFF / RAW_FS)
        self.phaser = np.exp(np.arange(1, RAW_FS + (overlap * 2) + 1) * phaser)
        del phaser
        b, a = signal.ellip(4, 1, 100, FC / (RAW_FS / 2), 'low', analog=False)
        self.filt = signal.dlti(b, a)
        # self.static_config(0.00007797103913, 9.14960812776426, 0.269693977605989, 2, 'low')
        self.config_flag = False

        # consumer and sender thread
        self.sender = Thread(target=self.consumer, args=[])

    def start_consuming(self):
        self.sender.start()

    def send_packet(self, info, chunk):
        if self.out_q.qsize() / QUEUE_SIZE < 1:
            self.out_q.put((info, chunk, time.time(), self.channel))
        else:
            self.out_q.put((info, None, len(chunk), self.channel))
        time.sleep(0.5 * (info[1] - info[0]) / FS)

    def consumer(self):
        startt = time.time()
        while True:
            if not self.in_q.empty():
                startt = time.time()
                buff = self.in_q.get()
                very_start = buff[1]
                
                if CHANNELIZE == True:
                	buff = channelize(buff[0], self.phaser, self.filt)
                else:
                	buff = buff[0].astype(np.complex64)
	        	
                buff = buff[10 * UPSAMPLE_FACTOR:-10 * UPSAMPLE_FACTOR]
                if not self.config_flag:
                    self.configNF(buff, 'low')
                    self.config_flag = True

                AS = self.Active_Sess_Detect2(buff, 'low', very_start)

                for sess in AS:
                    if sess[2] != 0:
                        if self.frag_buff.size == 0:
                            if sess[2] == 2:
                                # just send session, nothing to append to
                                # concat todo: This is a new feature if nothing ahead
                                self.last_buff = np.append(self.last_buff, buff[sess[0]:sess[1]])
                                self.send_packet(sess, self.last_buff)
                            else:
                                # add to frag_buff
                                self.frag_buff = np.append(self.frag_buff, buff[sess[0]:])
                                self.frag_store = sess
                        else:
                            if sess[2] == 2:
                                # concat
                                self.frag_store = weighted_sess_combine(self.frag_store, sess)
                                self.frag_buff = np.append(self.frag_buff, buff[sess[0]:sess[1]])

                                # send
                                self.send_packet(self.frag_store, self.frag_buff)

                                # clear
                                self.frag_store = None
                                self.frag_buff = np.ndarray((0,), dtype=np.complex64)
                            elif sess[2] == 1:
                                # send old session and replace with new
                                self.send_packet(self.frag_store, self.frag_buff)

                                self.frag_buff = buff[sess[0]:sess[1]]
                                self.frag_store = sess
                            else:
                                # concat and continue
                                self.frag_store = weighted_sess_combine(self.frag_store, sess)
                                self.frag_buff = np.append(self.frag_buff, buff[sess[0]:sess[1]])
                    else:
                        self.send_packet(sess, buff[sess[0]:sess[1]])
                self.last_buff = buff[-UPSAMPLE_FACTOR * 2 ** 12:]
            else:
                if time.time() - startt > 25:
                    return
                else:
                    time.sleep(0.25)

    # Active session detection
    # input a buffer to parse over (should be larger than 2^12 samples ideally)
    # input weather to dechirp with the high or low SF's
    # returns an array of active session values, these values contains:
    #   --> session start
    #   --> session end
    #   --> fragmented (1 = end of buff, 2 = start of buff, 3 = both, 0 means no fragmentation)
    #   --> Session magnitude
    #   --> Mini Histogram
    #   --> ddencmp values

    def Active_Sess_Detect2(self, x_1, level, tt):
        if level == 'high':
            Super_DC = self.ASD_big_superDC
            thresh_high = self.ASD_big_thresh_high
            thresh_low = self.ASD_big_thresh_low
            N = int(2 ** 12)
        else:
            Super_DC = self.ASD_small_superDC
            thresh_high = self.ASD_small_thresh_high
            thresh_low = self.ASD_small_thresh_low
            N = int(2 ** 9)

        win_jump = math.floor(N * UPSAMPLE_FACTOR / self.win_jump_factor)
        thresh_range = (thresh_high - thresh_low) / 4
        thresh = [thresh_high, thresh_high - thresh_range * 2, thresh_high - thresh_range * 3, thresh_low]

        pot_wind_start = 0

        PG_history = []
        uplink_wind = []
        store_pg = []
        store_var = []

        for i in range(math.floor(len(x_1) / win_jump) - self.win_jump_factor):  # loop over the samples
            wind = x_1[i * win_jump: i * win_jump + (N * UPSAMPLE_FACTOR)]

            # determine the peak gain here
            wind_fft = np.abs(np.fft.fft(wind * Super_DC))
            wind_fft = wind_fft[np.concatenate([np.arange(N // 2, dtype=int),
                                                np.arange((N // 2 + (UPSAMPLE_FACTOR - 1) * N),
                                                          UPSAMPLE_FACTOR * N, dtype=int)])]
            noise_floor = np.mean(wind_fft)
            tmp = np.partition(-wind_fft, self.num_keep)[:self.num_keep]
            fft_peak = np.sum(-tmp)
            peak_gain = 10 * math.log10(fft_peak / noise_floor)
            PG_history.append(-tmp / noise_floor)

            store_pg.append(peak_gain)

            if self.curr_state == 0:  # started a session
                store_var.append(thresh[self.curr_state])
                if peak_gain > thresh[self.curr_state]:
                    pot_wind_start = i
                    self.curr_state = 1
                else:
                    self.curr_state = 0
                    if len(PG_history) >= self.front_buf:  # make sure we don't record excessive ML data
                        PG_history.pop(0)
            elif self.curr_state == 1:
                store_var.append(thresh[self.curr_state])
                if peak_gain > thresh[self.curr_state]:
                    self.curr_state = 2
                else:
                    self.curr_state = 4
            elif self.curr_state == 2:
                store_var.append(thresh[self.curr_state])
                if peak_gain > thresh[self.curr_state]:
                    self.low_buffer = self.win_jump_factor * 4
                    self.curr_state = 3
                else:
                    self.curr_state = 1
            elif self.curr_state == 3:
                store_var.append(thresh[self.curr_state])
                if peak_gain > thresh[self.curr_state]:
                    self.low_buffer = self.win_jump_factor * 4
                    self.curr_state = 3
                else:
                    self.low_buffer -= 1
                    if self.low_buffer == 0:
                        self.curr_state = 6
            elif self.curr_state == 4:  # possible end to active session
                store_var.append(thresh[1])
                if peak_gain > thresh[1]:
                    self.curr_state = 2
                else:
                    self.curr_state = 5
            elif self.curr_state == 5:  # last chance
                store_var.append(thresh[1])
                if peak_gain > thresh[1]:
                    self.curr_state = 2
                else:
                    self.curr_state = 0
            elif self.curr_state == 6:  # store the session here
                store_var.append(thresh[0])
                pot_wind_end = i
                # todo: WE can just try to send right away here, no need to buffer every second, fragments might need to though!
                sess_mag = np.mean(np.abs(x_1[
                                          max((pot_wind_start - self.front_buf), 0) * win_jump:
                                          min((pot_wind_end + self.rear_buf) * win_jump,
                                              len(x_1))])) / self.Noise_Avg
                mini_hist = mini_hash_set(np.array(PG_history).flatten())
                dden = DWT.ret_dden(x_1[
                                    max((pot_wind_start - self.front_buf), 0) * win_jump:
                                    min((pot_wind_end + self.rear_buf) * win_jump, len(x_1))])

                uplink_wind.append(
                    [(pot_wind_start - self.front_buf), (pot_wind_end + self.rear_buf), 0, np.log10(sess_mag),
                     mini_hist, dden, tt])
                PG_history.clear()
                self.curr_state = 0

        # need to add fragmented flag for end
        if self.curr_state > 0:
            self.low_buffer = max(self.win_jump_factor * 2, self.low_buffer)
            self.curr_state = 3
            sess_mag = np.mean(
                np.abs(x_1[max((pot_wind_start - self.front_buf), 0) * win_jump:len(x_1)])) / self.Noise_Avg
            mini_hist = mini_hash_set(np.array(PG_history).flatten())
            dden = DWT.ret_dden(x_1[max((pot_wind_start - self.front_buf), 0) * win_jump:len(x_1)])
            uplink_wind.append(
                [(pot_wind_start - self.front_buf), int(len(x_1)), 1,
                 np.log10(sess_mag), mini_hist, dden, tt])

        if len(uplink_wind) > 0 and uplink_wind[0][0] <= 0:  # need to add fragmented flag
            uplink_wind[0][0] = 0
            uplink_wind[0][6] = 0
            uplink_wind[0][2] = uplink_wind[0][2] | 2

        small_arr = []
        for i in uplink_wind:
            if i[1] - i[0] > 10 or i[2] != 0:
                small_arr.append(i)
        uplink_wind = small_arr

        for i in uplink_wind:
            i[0] *= win_jump
            i[1] *= win_jump
            i[6] += (i[0] / 500000)
            if i[0] < 0:
                i[0] = 0
                i[6] = 0
            if i[1] > len(x_1):
                i[1] = len(x_1)

        # plt.plot(range(len(store_pg)), store_pg, label="Peak-Gains")
        # plt.plot(range(len(store_pg)), store_var, label="Thresh")
        # plt.legend()
        # plt.show()

        return uplink_wind

    def static_config(self, noise, thresh, std, mult, level):
        self.Noise_Avg = noise
        stdHigh = mult * std
        stdLow = (mult - 1) * std

        if level == 'high':
            self.ASD_big_thresh_high = thresh + stdHigh
            self.ASD_big_thresh_low = thresh + stdLow
        else:
            self.ASD_small_thresh_high = thresh + stdHigh
            self.ASD_small_thresh_low = thresh + stdLow

    # function to set up constants for initialization. Mostly searching for noise floor constants here
    def configNF(self, config_buffer, level):
        # config ASD_small
        if level == 'high':
            N = 2 ** 12
        else:
            N = 2 ** 9
        win_jump = math.floor(N * UPSAMPLE_FACTOR / self.win_jump_factor)
        Dechirper = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=(N * UPSAMPLE_FACTOR * 2)).view(
            np.complex128)
        recorder = []

        for i in range(math.floor(len(config_buffer) / win_jump) - self.win_jump_factor):
            wind = config_buffer[i * win_jump: i * win_jump + (N * UPSAMPLE_FACTOR)]
            wind_fft = np.abs(np.fft.fft(wind * Dechirper))
            wind_fft = wind_fft[np.concatenate([np.arange(N // 2, dtype=int),
                                                np.arange((N // 2 + (UPSAMPLE_FACTOR - 1) * N),
                                                          UPSAMPLE_FACTOR * N, dtype=int)])]
            noise_floor = np.mean(wind_fft)
            fft_peak = sum(-np.partition(-wind_fft, self.num_keep)[:self.num_keep])
            val = 10 * math.log10(fft_peak / noise_floor)
            recorder.append(val)

        std = np.std(recorder)
        stdHigh = 2.5 * std
        stdLow = 1.5 * std
        thresh = np.mean(recorder)

        self.Noise_Avg = np.mean(np.abs(config_buffer))

        print(self.Noise_Avg)
        print(thresh + stdHigh)
        print(thresh + stdLow)

        if level == 'high':
            self.ASD_big_thresh_high = thresh + stdHigh
            self.ASD_big_thresh_low = thresh + stdLow
        else:
            self.ASD_small_thresh_high = thresh + stdHigh
            self.ASD_small_thresh_low = thresh + stdLow
