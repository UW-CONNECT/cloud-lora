from Demodulators.Demod_Interface import Demod_Interface
import numpy as np
import math
from .decode import lora_decoder
from .DC_gen import DC_gen
#import matplotlib.pyplot as plt


class Std_LoRa(Demod_Interface):
    def __init__(self, num_preamble, num_sync, num_DC, num_data_sym, check_crc):
        self.num_preamble = num_preamble
        self.num_sync = num_sync
        self.num_DC = num_DC
        self.num_data_sym = num_data_sym
        self.check_crc = check_crc

        self.pkt_starts = []
        self.demod_sym = []

    def Evaluate(self, Rx, SF: int, BW: int, FS: int, PRINT: False):
        self.demodulate(Rx, SF, BW, FS)
        return self.decode(SF, PRINT)

    def decode(self, SF, PRINT):
        decoded_packets = 0
        final_data = []

        if len(self.demod_sym) > 0 and len(self.demod_sym[0]) > 0:
            for i, syms in enumerate(self.demod_sym):
                message = lora_decoder(np.mod(np.add(syms, -1), (2**SF)), SF, self.check_crc)
                if message is not None:
                    decoded_packets += 1
                    final_data.append((self.pkt_starts[i], self.demod_sym[i], message))
            final_data = self.remove_sim(final_data, 10 * (2 ** SF))

        self.demod_sym = []
        self.pkt_starts = []
        if PRINT:
            for d in final_data:
                print(d[2].tolist())
                #print(''.join([chr(int(c)) for c in d[2][7:]]))

        return final_data

    def demodulate(self, Rx, SF: int, BW: int, FS: int) -> list:
        self.pkt_starts = self.pkt_detection(Rx, SF, BW, FS, self.num_preamble)
        self.demod_sym = self.lora_demod(Rx, SF, BW, FS, self.num_preamble, self.num_sync, self.num_DC,
                                         self.num_data_sym, self.pkt_starts)
        return [0, 0]

    def pkt_detection(self, Rx_Buffer, SF, BW, FS, num_preamble):
        upsampling_factor = int(FS / BW)
        N = int(2 ** SF)
        num_preamble -= 1  # need to find n-1 total chirps (later filtered by sync word)

        DC_upsamp = DC_gen(SF, BW, FS)

        # Preamble Detection
        ind_buff = np.array([])
        count = 0
        Pream_ind = np.array([], int)

        loop = 0
        for off in range(3):
            offset = off * upsampling_factor * N // 3
            loop = Rx_Buffer.size // (upsampling_factor * N) - 1
            for i in range(loop):
                temp_wind_fft = abs(
                    np.fft.fft(Rx_Buffer[(i * upsampling_factor * N) + offset:
                                         ((i + 1) * upsampling_factor * N) + offset] * DC_upsamp, axis=0))
                temp_wind_fft_idx = np.concatenate(
                    [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])
                temp_wind_fft = temp_wind_fft[temp_wind_fft_idx]
                b = np.argmax(temp_wind_fft)
                if len(ind_buff) >= num_preamble:
                    ind_buff = ind_buff[-(num_preamble - 1):]
                    ind_buff = np.append(ind_buff, b)
                else:
                    ind_buff = np.append(ind_buff, b)

                if ((sum(abs(np.diff(np.mod(ind_buff, N + 1)))) <= (num_preamble + 4) or
                     sum(abs(np.diff(np.mod(ind_buff, N)))) <= (num_preamble + 4) or
                     sum(abs(np.diff(np.mod(ind_buff, N - 1)))) <= (num_preamble + 4)) and
                        ind_buff.size >= num_preamble - 1):
                    if np.sum(np.abs(Rx_Buffer[(i * upsampling_factor * N)
                                               + offset:((i + 1) * upsampling_factor * N) + offset])) != 0:
                        count = count + 1
                        Pream_ind = np.append(Pream_ind, (i - (num_preamble - 1)) * (upsampling_factor * N) + offset)

        # print('Found ', count, ' Preambles')
        if count >= (loop * 0.70):
            Preamble_ind = np.array([], int)
            return Preamble_ind

        # Synchronization
        Pream_ind.sort()
        shifts = np.arange(-N / 2, N / 2, dtype=int) * upsampling_factor
        new_pream = []
        for i in range(len(Pream_ind)):
            ind_arr = np.array([])

            for j in shifts:
                if Pream_ind[i] + j < 0:
                    ind_arr = np.append(ind_arr, -1)
                    continue

                temp_wind_fft = abs(
                    np.fft.fft(Rx_Buffer[(Pream_ind[i] + j): (Pream_ind[i] + j + upsampling_factor * N)] * DC_upsamp,
                               upsampling_factor * N, axis=0))
                temp_wind_fft = temp_wind_fft[np.concatenate(
                    [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])]
                b = temp_wind_fft.argmax()
                ind_arr = np.append(ind_arr, b)

            nz_arr = (ind_arr == 0).nonzero()
            if len(nz_arr) != 0:
                new_pream = new_pream + (shifts[nz_arr] + Pream_ind[i]).tolist()

        # sub-sample sync
        Pream_ind = new_pream
        shifts = np.arange(-upsampling_factor, upsampling_factor + 1, dtype=int)
        for i in range(len(Pream_ind)):
            amp_arr = []

            for j in shifts:
                if Pream_ind[i] + j < 0:
                    amp_arr.append([-1, j])
                    continue

                temp_wind_fft = abs(
                    np.fft.fft(Rx_Buffer[(Pream_ind[i] + j): (Pream_ind[i] + j + upsampling_factor * N)] * DC_upsamp,
                               upsampling_factor * N, axis=0))
                temp_wind_fft = temp_wind_fft[np.concatenate(
                    [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])]

                b = temp_wind_fft.argmax()
                if b == 0:
                    a = temp_wind_fft[0]
                    amp_arr.append([a, j])

            if len(amp_arr) != 0:
                Pream_ind[i] = Pream_ind[i] + max(amp_arr)[1]

        # SYNC WORD DETECTION
        count = 0
        Pream_ind = list(set(Pream_ind))
        Pream_ind.sort()
        Preamble_ind = np.array([], int)
        for i in range(len(Pream_ind)):
            if ((Pream_ind[i] + 9 * (upsampling_factor * N) > Rx_Buffer.size) or (
                    Pream_ind[i] + 10 * (upsampling_factor * N) > Rx_Buffer.size)):
                continue

            sync_wind1 = abs(np.fft.fft(Rx_Buffer[(Pream_ind[i] + 8 * upsampling_factor * N): (
                    Pream_ind[i] + 9 * upsampling_factor * N)] * DC_upsamp, axis=0))
            sync_wind2 = abs(np.fft.fft(Rx_Buffer[(Pream_ind[i] + 9 * upsampling_factor * N): (
                    Pream_ind[i] + 10 * upsampling_factor * N)] * DC_upsamp, axis=0))
            sync_wind1 = sync_wind1[np.concatenate(
                [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])]
            sync_wind2 = sync_wind2[np.concatenate(
                [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])]

            s1 = sync_wind1.argmax()
            s2 = sync_wind2.argmax()
            if s1 == 8 and s2 == 16:
                count = count + 1
                Preamble_ind = np.append(Preamble_ind, Pream_ind[i])

        return Preamble_ind

    def lora_demod(self, Rx_Buffer, SF, BW, FS, num_preamble, num_sync, num_DC, num_data_sym, Preamble_ind):
        upsampling_factor = int(FS / BW)
        N = int(2 ** SF)
        demod_sym = np.array([], int, ndmin=2)

        DC_upsamp = DC_gen(int(math.log2(N)), BW, FS)
        Data_frame_st = Preamble_ind + int((num_preamble + num_sync + num_DC) * N * upsampling_factor)

        for j in range(Preamble_ind.shape[0]):
            demod = np.empty((1, num_data_sym), int)
            for i in range(num_data_sym):
                if Data_frame_st[j] + (i + 1) * upsampling_factor * N > Rx_Buffer.size:
                    demod[:, i] = -1
                    continue

                temp_fft = abs(np.fft.fft(Rx_Buffer[(Data_frame_st[j] + i * upsampling_factor * N): (
                        Data_frame_st[j] + (i + 1) * upsampling_factor * N)] * DC_upsamp, axis=0))
                temp_fft = temp_fft[np.concatenate(
                    [np.arange(0, N // 2), np.arange(N // 2 + (upsampling_factor - 1) * N, upsampling_factor * N)])]

                b = temp_fft.argmax()
                demod[:, i] = b

            if j == 0:
                demod_sym = demod
            else:
                demod_sym = np.vstack((demod_sym, demod))

        demod_sym = demod_sym % N
        return demod_sym

    def remove_sim(self, vals, dist):
        if len(vals) <= 1:
            return vals
        ret = []
        curr_index = vals[0][0]
        ret.append(vals[0])
        for i, _ in enumerate(vals):
            if i == 0:
                continue
            else:
                if vals[i][0] - curr_index > dist:
                    ret.append(vals[i])
                    curr_index = vals[i][0]
        return ret
