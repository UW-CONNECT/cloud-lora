import socket
import multiprocessing

LOGGER_LOC = 'DATA_LOG.txt'
LORA_SF = [7, 8, 9]  # 12;            # LoRa spreading factor
LORA_BW = 125e3  # 125e3;        # LoRa bandwidth
FS = 500000  # recerver's sampling rate

# Receiving device parameters
NUM_PREAMBLE = 8  # down sampling factor
NUM_SYNC = 2
NUM_DC = 2.25
MAX_DATA_SYM = 100
SYNC1 = 8
SYNC2 = 16
HAS_CRC = True

# networking parameters
UPLINK_PORT = 65432
# MY_IP = socket.gethostbyname(socket.gethostname())
MY_IP = '0.0.0.0'

# processing parameters
NUM_CORES = 5

# (id, chan):(port, queue)
PORT_MAP = {
    (1, 1): (63201, multiprocessing.Queue()),
    (1, 2): (63202, multiprocessing.Queue()),
    (1, 3): (63203, multiprocessing.Queue()),
    (1, 4): (63204, multiprocessing.Queue()),
    (1, 5): (63205, multiprocessing.Queue()),
    (1, 6): (63206, multiprocessing.Queue()),
    (1, 7): (63207, multiprocessing.Queue()),
    (1, 8): (63208, multiprocessing.Queue())}
