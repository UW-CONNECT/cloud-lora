import numpy as np
import lz4.frame
import DWT_compress as DWT


class Active_Session:

    def __init__(self, radio_ID, channel, tag, start_time, thresh, buffer, fragmented=0, samp_size=32, Fs=125000):
        self.radio_ID = np.float64(radio_ID)  # ID for specific radio
        self.channel = np.float64(channel)  # specific channel
        self.pkt_num = np.float64(tag)  # tag for specific active session
        self.start_time = np.float64(start_time)  # Time of session arrival
        self.samp_size = np.float64(samp_size)  # Format (32, 16, possibly 8 in the future)
        self.fragmented = fragmented
        self.thresh = thresh
        self.buffer = buffer
        self.Fs = Fs

    def build_packet(self):
        ############## Perform a 5-level discrete wavelet decomposition #############
        multiplier = max(self.thresh, 0.0)
        # multiplier = 5
        (coefs, L) = DWT.DWT_compress(self.buffer, multiplier)

        if self.samp_size == 32.0:
            values = coefs.astype(dtype=np.float32)
        else:
            values = coefs.astype(dtype=np.float16)

        ############## active session info like its tag, ID and time ################
        sess_info = np.asarray([self.radio_ID, self.channel, self.pkt_num, self.start_time, self.samp_size,
                                self.fragmented, self.Fs, L[0], L[1], L[2], L[3], L[4], L[5]], dtype=np.float64)

        ################ build formatted file by concatting bytes then lz4 #####################
        fullpkt = sess_info.tobytes() + values.tobytes()
        lz4Vals = lz4.frame.compress(fullpkt, compression_level=16)
        lz4Vals = (len(lz4Vals)).to_bytes(4, 'big') + lz4Vals
        return lz4Vals

    @classmethod
    def from_packet(cls, packet):
        # First extract file metadata
        radio_ID = np.frombuffer(packet[0:8], dtype=np.float64)[0]  # each radio has an independent ID
        channel = np.frombuffer(packet[8:16], dtype=np.float64)[0]  # the channel of the active session
        pkt_num = np.frombuffer(packet[16:24], dtype=np.float64)[0]  # packet ID number (for RL)
        start_time = np.frombuffer(packet[24:32], dtype=np.float64)[0]  # when start of buffer was recorded
        samp_size = np.frombuffer(packet[32:40], dtype=np.float64)[0]  # format in 32 or 16 bit precision
        fragmented = np.frombuffer(packet[40:48], dtype=np.float64)[0]  # determine if needs to be appended
        Fs = np.frombuffer(packet[48:56], dtype=np.float64)[0]  # sampling frequency

        levels = np.frombuffer(packet[56:104],
                               dtype=np.float64)  # session score, offsets, and DWT level sizes
        if samp_size == 32.0:
            coefs = np.frombuffer(packet[104:], dtype=np.float32)
        else:
            coefs = np.frombuffer(packet[104:], dtype=np.float16)
            coefs = coefs.astype(dtype=np.float32)  # actual values

        levels = levels.astype(np.int64)
        buffer = DWT.DWT_rec(coefs, levels)

        ret = cls(radio_ID, channel, pkt_num, start_time, 0, buffer, fragmented, samp_size, Fs)
        return ret
