import socket

OS_CONF = 'Linux'
#OS_CONF = 'Windows'

if OS_CONF == 'Windows':
	FILE_PATH = 'C:/Users/danko/Desktop/RL_Data/Outdoor_test/'
else:
	FILE_PATH = '/home/uwcon/Desktop/RL_Data/'

################# LoRa and SDR info #########################
CENTER_FREQ = 903.3e6           		# SDR center frequency
RAW_FS = 500e3					# SDR's raw sampling freq
LORA_CHANNELS = [1 , 2, 3, 4, 5, 6, 7, 8]  # channels to process
CHANNELIZE = False				# weather to channelize locally or in gnuradio
FREQ_OFF = 200e3                		# channel size
FC = 70e3                       		# channel cutoff freq

UPSAMPLE_FACTOR = 4             		# factor to downsample to
FS = 125e3 * UPSAMPLE_FACTOR    		# sampling rate after conversion
BW = 125e3                      		# LoRa signal bandwidth
OVERLAP = int(10 * RAW_FS / BW)
THROTTLE_RATE = 1

SEND_SIZE = 32                  		# can either be 32 or 16 bit
############################################################


################# Network variables ########################
UPLINK_PORT = 65432
MY_IP = socket.gethostbyname(socket.gethostname())
SERVER_IP = '10.141.223.206'#socket.gethostbyname(socket.gethostname()) #'10.42.0.100'

sub_IP = MY_IP.split('.')
ID = 1
QUEUE_SIZE = 10 * min(len(LORA_CHANNELS), 4)
BACKUP_BW = 1000000000
############################################################


################# PPO variables ############################
SPE = 50            		# rounds of training before starting again
OBS_DIM = 12        		# dimension of input to MLP
NUM_ACT = 8					# number of possible actions to take per AP
ALPHA = 2					# first RL "Knob" where you can allow more/less latency

ACT_COMPRESS = True	 		# If user wants to compress with RL or with static DEFAULT_COMP
DEFAULT_COMP = 3		# If not using RL action, set static compression multiplier
LOAD_RL_FROM_SAVED = False	# Attempt to pre-load RL model
SAVE_RL = True				# Every 750 active periods, save model
############################################################

################ Function pointer to IQ stream #############
import IQ_source
IQ_SOURCE = IQ_source.UDP_load2
#IQ_SOURCE = IQ_source.file_load
############################################################
