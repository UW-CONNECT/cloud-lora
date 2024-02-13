import numpy as np
import scipy.signal


def DC_gen(SF, BW, Fs):
    sample_factor = int(Fs / BW)

    chirp_size = 2 ** SF
    symbol_length = sample_factor * chirp_size
    freq_shift = Fs / symbol_length
    symbol_time = 1 / freq_shift

    init_freq = -BW / 2
    final_freq = (BW / 2) - (freq_shift / sample_factor)
    t_arr = np.linspace(0, symbol_time - symbol_time/symbol_length, int(symbol_length))
    real = scipy.signal.chirp(t_arr, f0=init_freq,  f1=final_freq, t1=t_arr[-1], method='linear', phi=90)
    imag = scipy.signal.chirp(t_arr, f0=init_freq,  f1=final_freq, t1=t_arr[-1], method='linear', phi=180)

    return real + 1j * imag
