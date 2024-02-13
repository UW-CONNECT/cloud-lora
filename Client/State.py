# import numpy as np
# import math
# from client_config import *
#
#
# # class to store state of each active session, will have to keep updated here
# class State:
#     def __init__(self, mag, pg_hist, dden_val, BW_ratio, length, logits, action, comp_size, time_sent, normal_pkt_len,
#                  buff_size, last_BW, last_BW2, BW_obs):
#         self.mag = mag                  # normalized magnitude over noise floor
#         self.pg_hist = pg_hist          # peak gain histogram
#         self.dden_val = dden_val        # ddencmp value of active session
#         self.BW_ratio = BW_ratio        # might be more complex (rtt etc.. )
#         self.length = length
#         self.logits = logits
#         self.action = action
#         self.comp_size = comp_size
#         self.time_sent = time_sent
#         self.normal_pkt_len = normal_pkt_len
#         self.buff_size = buff_size
#         self.last_BW = last_BW
#         self.last_BW2 = last_BW2
#         self.BW_obs = BW_obs
#
#         self.post_buff_size = buff_size
#         self.latency = -1
#
#     def get_obs(self):
#         return np.array([self.BW_obs, self.BW_ratio, self.last_BW, self.last_BW2, self.normal_pkt_len, self.buff_size,
#                          self.mag, self.dden_val,] + self.pg_hist, dtype=np.float32)
#
#     def get_string(self):
#         return f"{self.last_BW}\t{self.last_BW2}\t{self.BW_ratio}\t{self.BW_obs}\t{self.buff_size}\t{self.post_buff_size}" \
#                f"\t{self.latency}\t{self.mag}\t{self.pg_hist[0]}\t{self.length}\t{self.comp_size}\t" \
#                f"{self.action[0]}"
#
#     def calc_rewards(self, num_dec, success_rate):
#         rewards = 0
#         BW_Weight = 0.5
#         BW_cutoff = 0.9
#
#         LAT_Weight = 0.5
#         LAT_cutoff = 0.9
#
#         # if packet dropped return big negative
#         if num_dec[0] == -1:
#             return -10
#
#         # weight reward by keeping within BW constraints
#         weight = self.BW_ratio	# todo: this can be multiplied by number of channels to maintain fairness
#         if weight > BW_cutoff:
#             if weight > BW_cutoff + 0.2:
#                 rewards -= BW_Weight
#             else:
#                 rewards -= (weight - BW_cutoff) * 5 * BW_Weight
#
#         # if latency is beyond alpha, use hinge loss
#         Comp_Lat = (8 * self.comp_size / (50000000 * math.exp(self.BW_obs * 3))) + self.latency
#         if Comp_Lat > LAT_cutoff * ALPHA:
#             if Comp_Lat > LAT_cutoff + 0.2:
#                 rewards -= LAT_Weight
#             else:
#                 rewards -= (Comp_Lat - LAT_cutoff) * 5 * LAT_Weight
#
#         # bread and butter reward here, if we decode we reward
#         if success_rate < 0.1:
#             success_rate = 0.1
#         Dec_rate = weight / success_rate
#         for i, _ in enumerate(num_dec):
#             if num_dec[i] > 0:
#                 rewards += Dec_rate * num_dec[i]
#         return rewards

import numpy as np
import math
from utils import *
from client_config import *


# class to store state of each active session, will have to keep updated here
class State:
    def __init__(self, mag, pg_hist, dden_val, BW_ratio, length, logits, action, comp_size, start_time, time_sent,
                 normal_pkt_len,
                 buff_size, last_BW, last_BW2, BW_obs):
        self.mag = mag  # normalized magnitude over noise floor
        self.pg_hist = pg_hist  # peak gain histogram
        self.dden_val = dden_val  # ddencmp value of active session
        self.BW_ratio = BW_ratio  # might be more complex (rtt etc.. )
        self.length = length
        self.logits = logits
        self.action = action
        self.comp_size = comp_size
        self.start_time = start_time
        self.time_sent = time_sent
        self.normal_pkt_len = normal_pkt_len
        self.buff_size = buff_size
        self.last_BW = last_BW
        self.last_BW2 = last_BW2
        self.BW_obs = BW_obs
        self.latency = -1

    def get_obs(self):
        return np.array([self.BW_obs, self.BW_ratio, self.last_BW, self.last_BW2, self.normal_pkt_len, self.buff_size,
                         self.mag, self.dden_val] + self.pg_hist, dtype=np.float32)

    def get_string(self):
        return f"{self.last_BW}\t{self.last_BW2}\t{self.BW_ratio}\t{self.BW_obs}\t{self.buff_size}\t{self.start_time}" \
               f"\t{self.latency}\t{self.mag}\t{self.pg_hist[0]}\t{self.length}\t{self.comp_size}\t" \
               f"{self.action[0]}"

    def calc_rewards(self, num_dec, success_rate):
        rewards = 0
        BW_Weight = 0.5
        BW_cutoff = 0.9

        LAT_Weight = 0.5
        LAT_cutoff = 0.9

        # if packet dropped return big negative
        if num_dec[0] == -1:
            return -10

        # weight reward by keeping within BW constraints
        weight = self.BW_ratio
        if weight > BW_cutoff:
            if weight > BW_cutoff + 0.2:
                rewards -= BW_Weight
            else:
                rewards -= (weight - BW_cutoff) * 5 * BW_Weight

        # if latency is beyond alpha, use hinge loss
        Comp_Lat = (8 * self.comp_size / (50000000 * math.exp(self.BW_obs * 3))) + self.latency + RTT_sim()
        if Comp_Lat > LAT_cutoff * ALPHA:
            if Comp_Lat > (LAT_cutoff * ALPHA) + 0.2:
                rewards -= LAT_Weight
            else:
                rewards -= (Comp_Lat - (LAT_cutoff * ALPHA)) * 5 * LAT_Weight

        # bread and butter reward here, if we decode we reward
        if success_rate < 0.01:
            success_rate = 0.01
        Dec_rate = 1 / success_rate
        for i, _ in enumerate(num_dec):
            if num_dec[i] > 0:
                rewards += Dec_rate * num_dec[i]
        return rewards
