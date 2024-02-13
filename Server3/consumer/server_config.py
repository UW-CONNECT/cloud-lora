import socket

LOGGER_LOC = 'DATA_LOG.txt'
LORA_SF = [7, 8, 9]  # 12;            # LoRa spreading factor
LORA_BW = 125e3  # 125e3;        # LoRa bandwidth
FS = 500000  # recerver's sampling rate

# Receiving device parameters
NUM_PREAMBLE = 8  # down sampling factor
NUM_SYNC = 2
NUM_DC = 2.25
MAX_DATA_SYM = 50
SYNC1 = 8
SYNC2 = 16
HAS_CRC = True

# networking parameters
UPLINK_PORT = 65432
# MY_IP = '10.42.0.155'
# MY_IP = '192.168.1.3'
# MY_IP = '10.140.45.214'
MY_IP = socket.gethostbyname(socket.gethostname())
# MY_IP = '0.0.0.0'

# processing parameters
NUM_CORES = 5
