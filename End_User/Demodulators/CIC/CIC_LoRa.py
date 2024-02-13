from .DC_location_correlation import DC_location_correlation
from .UC_location_corr_DC_based import UC_location_corr_DC_based
from .dnsamp_buff import dnsamp_buff
from .filter_false_postives import filter_false_postives
from .active_sess_dechirp import active_sess_dechirp
from .active_sess_split import active_sess_split
from .param_configs import param_configs
from .CIC_Demod import CIC_Demod
from Demodulators.Demod_Interface import Demod_Interface
import numpy as np
import time


class CIC_LoRa(Demod_Interface):
    def __init__(self, num_preamble, num_sync, num_DC, num_data_sym):
        self.num_preamble = num_preamble
        self.num_sync = num_sync
        self.num_DC = num_DC
        self.num_data_sym = num_data_sym

    def demodulate(self, Rx, SF: int, BW: int, Fs: int) -> list:
        global_start = time.time()

        ## Loading variables
        N = int(2**SF)
        upsampling_factor = int(Fs/BW)

        # LORA pkt variables
        num_preamble = param_configs(4)
        num_sync = param_configs(5)
        num_DC = param_configs(6)
        num_data_sym = param_configs(7)
        pkt_len = num_preamble + num_sync + num_DC + num_data_sym
        num_samples = pkt_len * N

        x_1 = Rx
        uplink_wind = active_sess_dechirp(x_1, SF, BW, Fs) - 1
        uplink_wind = active_sess_split(uplink_wind, 10*num_samples*upsampling_factor, 2.5*upsampling_factor*N)    # split up sessions longer than 10 times packet length

        demod_sym_stack = []
        Peaks = []
        for m in range(uplink_wind.shape[0]):
            # DC correlations to find LoRa pkts out of collision
            temp_buff = x_1[int(uplink_wind[m, 0]) : int(uplink_wind[m, 1]) + 1]
            temp_buff = temp_buff[:(len(temp_buff)//upsampling_factor)*upsampling_factor]
            DC_ind = DC_location_correlation(temp_buff[::upsampling_factor], SF)
            if DC_ind.shape[0] == 0:
                continue

            if (DC_ind[-1, 0] * upsampling_factor + ((num_DC + num_data_sym) * N * upsampling_factor)) > len(temp_buff):
                #  if a data portion of a packet is split then account for the length difference
                ex_samp = int((DC_ind[-1,0]*upsampling_factor + ((num_DC + num_data_sym)*N*upsampling_factor)) - len(temp_buff))
                temp_buff = x_1[uplink_wind[m,0] : uplink_wind[m,1] + ex_samp]
                temp_buff = temp_buff[0:int(np.floor(temp_buff.shape[0]/upsampling_factor)*upsampling_factor)]

            # UC correlation to filter false positives and frequency offset & Packets' SNR estimation
            # All possible downsampled Buffers with different starting sample for downsampling
            Rx_Buff_dnsamp = []
            for i in range(upsampling_factor):
                Rx_Buff_dnsamp.append(temp_buff[i::upsampling_factor])
            Rx_Buff_dnsamp = np.array(Rx_Buff_dnsamp)
            [Upchirp_ind] = UC_location_corr_DC_based(temp_buff[::upsampling_factor], DC_ind, SF, BW, Fs)
            if Upchirp_ind.shape[0] == 0:
                continue

            # for each Preamble detected, Choosing the correct downsampled buffer with any frequency offset
            # been compensated and determining Preamble Peak heights to be used later for Power filtering
            [Data_freq_off, Peak, Upchirp_ind, FFO] = dnsamp_buff(Rx_Buff_dnsamp, Upchirp_ind, SF)
            if Upchirp_ind.shape[0] == 0:
                continue

            # Filter False Positives based on 2-SYNC Words detected
            [Preamble_ind, bin_offsets, Data_out, Peak_amp, FFO] = filter_false_postives(Data_freq_off, Upchirp_ind, Peak, FFO, SF)
            if Preamble_ind.shape[0] == 0:
                continue

            # filter preambles that are with in 5 samples (same pkt detected twice due to Correlation peak energy spread)
            temp = []
            temp_data = []
            temp_peaks = []
            indices = np.vstack([[np.zeros(num_preamble)], Preamble_ind])
            Data = np.vstack([[np.zeros(Data_out.shape[1])], Data_out])
            peaks = np.vstack([[np.zeros(Peak_amp.shape[1])], Peak_amp])
            for i in range(1, indices.shape[0]):
                if len(temp) == 0:
                    temp.append(indices[i, :])
                    temp_data.append(Data[i, :])
                    temp_peaks.append(peaks[i, :])
                else:
                    if min(abs(indices[np.unravel_index(i, indices.shape, 'F')] - np.array(temp)[:, 1])) > 10:
                        temp.append(indices[i, :])
                        temp_data.append(Data[i, :])
                        temp_peaks.append(peaks[i, :])
            Pream_ind = np.array(temp)
            Data_out = np.array(temp_data)
            Peak_amp = np.array(temp_peaks)

            ##  Data Demodulation using CIC
            demod_sym = []
            for j in range(Pream_ind.shape[0]):
                demod_sym.append(CIC_Demod(Pream_ind[j, :], Data_out[j, :], Pream_ind, Peak_amp[j, :], j, SF, num_data_sym))
                demod_sym[j] = np.mod(demod_sym[j] + bin_offsets[j] - 2, N)
                demod_sym_stack.append(demod_sym[j])
            Peaks.append(Peak_amp)

        # Saving symbols and peaks to disk
        demod_sym_stack = np.array(demod_sym_stack)

        global_end = time.time()
        print(f'Demodulator: CIC; Time taken = {global_end - global_start} seconds')
        return demod_sym_stack.astype(int)+1
