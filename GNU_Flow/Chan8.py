#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: mos26
# GNU Radio version: 3.9.4.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import network
from gnuradio import uhd
import time



from gnuradio import qtgui

class Chan8(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "Chan8")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2e6
        self.dec_factor = dec_factor = 4
        self.M_PI = M_PI = 3.14159265358979323846
        self.FFtaps = FFtaps = [8.98440657102068e-05, 9.54644212279418e-05, 0.000153599440344735, 9.54644212279418e-05, 8.98440657102070e-05]
        self.FBtaps = FBtaps = [1, -3.74412117687482, 5.30709175926160, -3.37351017499529, 0.811127773099310]

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(903.2e6, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_bandwidth(samp_rate, 0)
        self.uhd_usrp_source_0.set_gain(50, 0)
        self.rational_resampler_xxx_0_0_0_6 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0_5 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0_4 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0_3 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0_1 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=dec_factor,
                taps=[],
                fractional_bw=0)
        self.network_udp_sink_0_0_1_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 4900, 0, 32768, False)
        self.network_udp_sink_0_0_1_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 5000, 0, 32768, False)
        self.network_udp_sink_0_0_1 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 5100, 0, 32768, False)
        self.network_udp_sink_0_0_0_0_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2030, 0, 32768, False)
        self.network_udp_sink_0_0_0_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2020, 0, 32768, False)
        self.network_udp_sink_0_0_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 1998, 0, 32768, False)
        self.network_udp_sink_0_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 2010, 0, 32768, False)
        self.network_udp_sink_0_0 = network.udp_sink(gr.sizeof_gr_complex, 1, '127.0.0.1', 5200, 0, 32768, False)
        self.iir_filter_xxx_0_0_2_0 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_2 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_1_1_0 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_1_1 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_1_0 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_1 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0_0 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.iir_filter_xxx_0_0 = filter.iir_filter_ccd(FFtaps, FBtaps, False)
        self.blocks_rotator_cc_0_0_1_0 = blocks.rotator_cc((-7 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0_1 = blocks.rotator_cc((5 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0_0_1_0 = blocks.rotator_cc((-5 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0_0_1 = blocks.rotator_cc((7 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0_0_0 = blocks.rotator_cc((-3 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0_0 = blocks.rotator_cc((3 * M_PI) / 10, False)
        self.blocks_rotator_cc_0_0 = blocks.rotator_cc((1 * M_PI) / 10, False)
        self.blocks_rotator_cc_0 = blocks.rotator_cc((-1 * M_PI) / 10, False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/media/uwcon/Extreme SSD/RealTimeData/Outdoor_12_28/1hr_15mbit_testing', False)
        self.blocks_file_sink_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_rotator_cc_0, 0), (self.iir_filter_xxx_0_0_0, 0))
        self.connect((self.blocks_rotator_cc_0_0, 0), (self.iir_filter_xxx_0_0, 0))
        self.connect((self.blocks_rotator_cc_0_0_0, 0), (self.iir_filter_xxx_0_0_1, 0))
        self.connect((self.blocks_rotator_cc_0_0_0_0, 0), (self.iir_filter_xxx_0_0_1_0, 0))
        self.connect((self.blocks_rotator_cc_0_0_0_1, 0), (self.iir_filter_xxx_0_0_1_1, 0))
        self.connect((self.blocks_rotator_cc_0_0_0_1_0, 0), (self.iir_filter_xxx_0_0_1_1_0, 0))
        self.connect((self.blocks_rotator_cc_0_0_1, 0), (self.iir_filter_xxx_0_0_2, 0))
        self.connect((self.blocks_rotator_cc_0_0_1_0, 0), (self.iir_filter_xxx_0_0_2_0, 0))
        self.connect((self.iir_filter_xxx_0_0, 0), (self.rational_resampler_xxx_0_0_0_3, 0))
        self.connect((self.iir_filter_xxx_0_0_0, 0), (self.rational_resampler_xxx_0_0_0_0, 0))
        self.connect((self.iir_filter_xxx_0_0_1, 0), (self.rational_resampler_xxx_0_0_0_4, 0))
        self.connect((self.iir_filter_xxx_0_0_1_0, 0), (self.rational_resampler_xxx_0_0_0, 0))
        self.connect((self.iir_filter_xxx_0_0_1_1, 0), (self.rational_resampler_xxx_0_0_0_6, 0))
        self.connect((self.iir_filter_xxx_0_0_1_1_0, 0), (self.rational_resampler_xxx_0_0, 0))
        self.connect((self.iir_filter_xxx_0_0_2, 0), (self.rational_resampler_xxx_0_0_0_5, 0))
        self.connect((self.iir_filter_xxx_0_0_2_0, 0), (self.rational_resampler_xxx_0_0_0_1, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.network_udp_sink_0_0_0_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0, 0), (self.network_udp_sink_0_0_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_0, 0), (self.network_udp_sink_0_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_1, 0), (self.network_udp_sink_0_0_0_0_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_3, 0), (self.network_udp_sink_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_4, 0), (self.network_udp_sink_0_0_1, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_5, 0), (self.network_udp_sink_0_0_1_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0_6, 0), (self.network_udp_sink_0_0_1_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_0_1, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_0_1_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_1, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_rotator_cc_0_0_1_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Chan8")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_bandwidth(self.samp_rate, 0)

    def get_dec_factor(self):
        return self.dec_factor

    def set_dec_factor(self, dec_factor):
        self.dec_factor = dec_factor

    def get_M_PI(self):
        return self.M_PI

    def set_M_PI(self, M_PI):
        self.M_PI = M_PI
        self.blocks_rotator_cc_0.set_phase_inc((-1 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0.set_phase_inc((1 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_0.set_phase_inc((3 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_0_0.set_phase_inc((-3 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_0_1.set_phase_inc((7 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_0_1_0.set_phase_inc((-5 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_1.set_phase_inc((5 * self.M_PI) / 10)
        self.blocks_rotator_cc_0_0_1_0.set_phase_inc((-7 * self.M_PI) / 10)

    def get_FFtaps(self):
        return self.FFtaps

    def set_FFtaps(self, FFtaps):
        self.FFtaps = FFtaps
        self.iir_filter_xxx_0_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_1.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_1_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_2.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_2_0.set_taps(self.FFtaps, self.FBtaps)

    def get_FBtaps(self):
        return self.FBtaps

    def set_FBtaps(self, FBtaps):
        self.FBtaps = FBtaps
        self.iir_filter_xxx_0_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_1.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_1_1_0.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_2.set_taps(self.FFtaps, self.FBtaps)
        self.iir_filter_xxx_0_0_2_0.set_taps(self.FFtaps, self.FBtaps)




def main(top_block_cls=Chan8, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
