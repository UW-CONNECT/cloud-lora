import math
import multiprocessing
import time
from threading import Thread
from queue import Queue

from matplotlib import pyplot as plt

from Active_Session import Active_Session
from Active_Period_Detector import Active_Period_Detector2
from State import State
from PPO import PPO
from utils import *

channel_streams = []
big_q = multiprocessing.Queue()
comp_q = multiprocessing.Queue(maxsize=1)
out_q = multiprocessing.Queue(maxsize=2)

# running average of BW usage and active period false positive rates
C_FP_rate = {}  # store False Positive rates channel-wise
AGG_FP_rate = []  # store false positive rate among all channels
BW_hist = []  # Store cumulative BW usage

# save directory info
model_dir = ''
save_dir = ''
if OS_CONF == 'Windows':
    save_dir = os.getcwd() + '\\Data\\' + str(LORA_CHANNELS[0])
    model_dir = os.getcwd() + '\\Models\\' + str(LORA_CHANNELS[0])
else:
    save_dir = os.getcwd() + '/Data/' + str(LORA_CHANNELS[0])
    model_dir = os.getcwd() + '/Models/' + str(LORA_CHANNELS[0])

try:
    os.mkdir(save_dir)
except FileExistsError:
    pass
try:
    os.mkdir(model_dir)
except FileExistsError:
    pass

if OS_CONF == 'Windows':
    save_dir += '\\'
    model_dir += '\\'
else:
    save_dir += '/'
    model_dir += '/'

# RL objects
my_ppo = PPO(OBS_DIM, NUM_ACT, SPE, model_dir, from_saved=LOAD_RL_FROM_SAVED)
episode_return, episode_length = 0, 0

# Packet management for RL
pkt_counter = 0
rew_counter = 0
Rewards = {}
History = {}


# create a method that will basically just spawn a Channel_Worker object and pass a multiprocessing queue
def spawn_a_worker(my_channel, input_queue, output_queue):
    worker = Active_Period_Detector2(my_channel, input_queue, output_queue)
    worker.start_consuming()


# a thread to manage every outgoing packet, we rely on a single socket to avoid BBR inter-flow contention
def uplink_manager():
    while True:
        if not big_q.empty():
            act_and_send(big_q.get(), big_q.qsize())
        else:
            time.sleep(0.05)


# Function to determine an action for a given active period, and convert it into a sendable form
#   in:     sess is the session information
#   out:    an active session object
def act_and_send(Pkt_To_Send, queue_size):
    global History, pkt_counter, my_ppo, comp_q

    info = Pkt_To_Send[0]
    buffer = Pkt_To_Send[1]
    start_t = info[6]
    channel = Pkt_To_Send[3]

    # organize info
    mag = info[3]
    hist = info[4].tolist()
    dden = info[5]
    pkt_size = info[1] - info[0]
    pkt_size_norm = pkt_size / FS
    buff_size = queue_size / QUEUE_SIZE
    act_time = time.time()

    # format and normalize bandwidth properties
    trim_by_time(act_time, BW_hist, 10)
    bandwid = BW_sim()  # bandwidth in bps not Bps
    BW_obs = math.log(bandwid / 5000000) / 3

    BW_possible, BW_used = integrate_BW(BW_hist, act_time, -1)
    BW_ratio = (BW_used * 8) / BW_possible
    BW_possible, BW_used = integrate_BW(BW_hist, act_time, 5)
    last_BW = (((BW_used * 8) / BW_possible) / 5)
    BW_possible, BW_used = integrate_BW(BW_hist, act_time, 10)
    last_BW2 = (((BW_used * 8) / BW_possible) / 5)

    # from session information (state) determine action
    observation = np.array([BW_obs, BW_ratio, last_BW, last_BW2, pkt_size_norm, buff_size, mag, dden] +
                           hist, dtype=np.float32)
    observation = observation.reshape(1, -1)
    logits, act = my_ppo.sample_action(observation)
    action = dden * DEFAULT_COMP * 5
    if ACT_COMPRESS:
        action = dden * act[0].numpy() * 5

    # create active session object
    zz = Active_Session(ID, channel, pkt_counter, start_t, action, buffer, info[2], SEND_SIZE, FS)
    state = State(mag, hist, dden, BW_ratio, pkt_size, logits, act, 0, start_t, act_time, pkt_size_norm,
                  buff_size, last_BW, last_BW2, BW_obs)

    History[pkt_counter] = state

    if buffer is not None:
        comp_q.put(zz)
    else:
        print("packet actually dropped, buffer was too full")
        Rewards[pkt_counter] = ([-1, -1, -1, -1, -1, -1], channel)
    pkt_counter += 1


def comp_send(data_in, data_out):
    while True:
        if not data_in.empty():
            pkt = data_in.get()
            data_out.put((pkt.build_packet(), pkt.pkt_num, pkt.channel))
        else:
            time.sleep(0.1)


# final stage to send packet, a single buffer allows for semi-blocking, this gives the manager time to compress etc..
def sending_queue(server_socket):
    global History, out_q, BW_hist, Rewards
    while True:
        if not out_q.empty():
            pkt, pkt_num, channel = out_q.get()
            print(f"Sending Packet: {pkt_num}")
            bandwid = 5000000 * math.exp(History[pkt_num].BW_obs * 3)
            History[pkt_num].comp_size = len(pkt)
            History[pkt_num].latency = min(time.time() - History[pkt_num].start_time, 10)
            History[pkt_num].time_sent = time.time()
            if (len(pkt) * 8) / bandwid > ALPHA / 2:
                print("packet actually dropped due to latency limitations")
                Rewards[pkt_num] = ([-1, -1, -1, -1, -1, -1], channel)
            else:
                BW_hist.append([History[pkt_num].time_sent, len(pkt), bandwid])
                try:
                    server_socket.send(bytes(pkt))
                except Exception as e:
                    print("Failed to send to server, either lost connection, or BW constrained, try again")
        else:
            time.sleep(0.05)


# we also receive rewards on this socket, feedback is sent to a dictionary for ordering and processing
def reward_manager(server_socket):
    global Rewards

    while True:
        full_msg = b''
        new_msg = True
        msglen = 0

        while True:
            full_msg += server_socket.recv(16)
            if new_msg:
                msglen = int.from_bytes(full_msg[:4], 'little')
                new_msg = False

            if len(full_msg) >= msglen + 4:
                pkt_index = int.from_bytes(full_msg[4:8], 'little')
                channel_num = int.from_bytes(full_msg[8:12], 'little', signed=True)
                num_dec = int.from_bytes(full_msg[12:16], 'little', signed=True)
                tmp = [0, 0, 0, 0, 0]
                tmp.append(num_dec)
                num_dec = np.array(tmp, dtype=np.int32)

                # add to reward dictionary for processing later on
                Rewards[pkt_index] = (num_dec, channel_num)

                full_msg = full_msg[msglen + 4:]
                new_msg = True


# Thread to process rewards IN ORDER, rewards processed out of order showed poor results
def process_rewards():
    global rew_counter, Rewards, History, my_ppo, episode_length, episode_return, C_FP_rate
    epoch_save_cnt = 30
    while True:
        if rew_counter in Rewards:
            print(f"Reward Counter: {rew_counter}")

            # get history to process
            hist = History.pop(rew_counter)
            num_dec = Rewards.pop(rew_counter)
            chann = num_dec[1]
            num_dec = num_dec[0]
            obs = hist.get_obs()
            obs = obs.reshape(1, -1)

            # keep tabs on false positive rates of APs
            # C_FP_rate[chann].append(max(sum(num_dec), 0))
            # if len(C_FP_rate[chann]) > 100:
            #     C_FP_rate[chann].pop(0)
            # success_rates = get_prev_success(C_FP_rate[chann])

            AGG_FP_rate.append(max(sum(num_dec), 0))
            if len(AGG_FP_rate) > 100:
                AGG_FP_rate.pop(0)
            success_rates = get_prev_success(AGG_FP_rate)

            # store observations internally to PPO agent
            reward = hist.calc_rewards(num_dec, success_rates)
            episode_return += reward
            episode_length += 1
            my_ppo.store_step(obs, hist.logits, hist.action, reward)

            # record information for later analysis
            store_vals = f"{chann}\t" + hist.get_string() + f"\t{reward}\t{sum(num_dec)}"
            store_results(save_dir + 'Rewards.txt', store_vals)

            # after each epoch, train the RL models
            if episode_length == SPE:
                # store_results('Episode_rewards.txt', self.episode_return)
                last_value = 0
                my_ppo.critic(obs.reshape(1, -1))
                my_ppo.buffer.finish_trajectory(last_value)
                my_ppo.train_self()
                episode_length = 0
                episode_return = 0
                epoch_save_cnt -= 1
                if SAVE_RL and epoch_save_cnt == 0:
                    print("saving RL...")
                    my_ppo.save()
                    epoch_save_cnt = 30

            rew_counter += 1
        else:
            time.sleep(0.25)


if __name__ == "__main__":
    for i in LORA_CHANNELS:
        in_queue = multiprocessing.Queue()
        channel_streams.append(in_queue)
        C_FP_rate[i] = []
        multiprocessing.Process(target=spawn_a_worker, args=(i, in_queue, big_q)).start()

    flow = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        flow.connect((SERVER_IP, UPLINK_PORT))
        print("Connected to server")
        print(QUEUE_SIZE)
    except Exception as e:
        print("Failed to connect to server")

    a = Thread(target=uplink_manager, args=[])
    b = Thread(target=sending_queue, args=[flow])
    c = Thread(target=reward_manager, args=[flow])
    d = Thread(target=process_rewards, args=[])
    a.setDaemon(True)
    b.setDaemon(True)
    c.setDaemon(True)
    d.setDaemon(True)
    a.start()
    b.start()
    c.start()
    d.start()

    myPool = multiprocessing.Pool(1, comp_send, (comp_q, out_q))

    time.sleep(2.0)
    for i in range(len(LORA_CHANNELS)):
        print(LORA_CHANNELS[i])
        multiprocessing.Process(target=IQ_SOURCE, args=(channel_streams[i], LORA_CHANNELS[i])).start()
    #while True:
    #    cc = 1
    time.sleep(7260)

    # time.sleep(2.0)
    # thread_arr = list()
    # for i in range(len(LORA_CHANNELS)):
    #     print(range(len(LORA_CHANNELS)))
    #     print(LORA_CHANNELS[i])
    #     thread_arr.append( Thread(target=IQ_SOURCE, args=[channel_streams[i], LORA_CHANNELS[i]]) )
    #     thread_arr[len(thread_arr) - 1].setDaemon(True)
    #     thread_arr[len(thread_arr) - 1].start()
    # while True:
    #     cc = 1
    #IQ_SOURCE(channel_streams)
    # time.sleep(20)

    myPool.terminate()
    myPool.join()
    flow.close()
