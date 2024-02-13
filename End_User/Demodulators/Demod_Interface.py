"""
Informal interface definition. Both individual input streams, and coherent combined input streams should be input here:
input:  Rx --> list of buffers of complex samples to process (buffers are of same length)
        SF --> Spreading factor of packets to demodulate
        BW --> Bandwidth of packets to demodulate
        FS --> Sample rate of input stream

Return: tuple containing packet starts, and list of lists of each packets symbols
"""


class Demod_Interface:
    def demodulate(self, Rx, SF: int, BW: int, FS: int) -> list:
        """Process input stream here"""
        pass
