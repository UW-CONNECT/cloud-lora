import socket

LORA_SF = [7, 8, 9]  # 12;            # LoRa spreading factor
LORA_BW = 125e3  # 125e3;        # LoRa bandwidth

# Receiving device parameters
NUM_PREAMBLE = 8  # down sampling factor
NUM_SYNC = 2
NUM_DC = 2.25
MAX_DATA_SYM = 100
SYNC1 = 8
SYNC2 = 16
HAS_CRC = True

SERVER_IP = '127.0.0.1'
#SERVER_IP = '20.228.167.24'
