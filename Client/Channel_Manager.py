from queue import Queue
from threading import Thread
import time
import math
from Active_Period_Detector import Active_Period_Detector
from PPO import PPO
from Active_Session import Active_Session
from State import State
from utils import *
from client_config import *

"""
Channel Worker pipeline:
[stream]: reads from input queue, performs APD and places into sending queue
[sender]: if connection established, sender will read from sending queue and send via TCP to cloud
[Receiver]: waits for incoming rewards from the socket, places them into the reward organizer
[Rewarder]: after rewards are returned in order we can train the RL occasionally

input[] -> stream -> send_queue[] -> sender -> CLOUD
CLOUD -> receiver -> received_buffer[] -> rewarder
"""


class Channel_Manager:
    def __init__(self, channel, input_stream, output_stream, reward_stream):
        # internal data to save
        print(f"Hey I am channel: {channel}")
        self.channel = channel
        self.center_freq = CENTER_FREQ + channel * FREQ_OFF
        overlap = 10 * RAW_FS / BW
        phaser = 1j * 2 * np.pi * self.channel * (FREQ_OFF / RAW_FS)
        self.phaser = np.exp(np.arange(1, RAW_FS + (overlap * 2) + 1) * phaser)
        del phaser

        b, a = signal.ellip(4, 1, 100, FC / (RAW_FS / 2), 'low', analog=False)
        self.filt = signal.dlti(b, a)
        
        if OS_CONF == 'Windows':
            self.save_dir = os.getcwd() + '\\Data\\' + str(channel)
            self.model_dir = os.getcwd() + '\\Models\\' + str(channel)
        else:
            self.save_dir = os.getcwd() + '/Data/' + str(channel)
            self.model_dir = os.getcwd() + '/Models/' + str(channel)
            
        try:
            os.mkdir(self.save_dir)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.model_dir)
        except FileExistsError:
            pass
            
        if OS_CONF == 'Windows':
            self.save_dir += '\\'
            self.model_dir += '\\'
        else:
            self.save_dir += '/'
            self.model_dir += '/'

        # History of packets sent to cloud and reward ordering buffer
        self.pkt_counter = 0
        self.rew_counter = 0
        self.Rewards = {}
        self.History = {}

        # Queues to send and receive packets
        self.input = input_stream
        self.output = output_stream
        self.rew_queue = reward_stream
        self.apd_queue = Queue()
        self.send_queue = Queue()

        # running average of BW usage and active period false positive rates
        self.Prev_Send = []
        self.Prev_send_rates = []

        # RL objects
        self.my_ppo = PPO(OBS_DIM, NUM_ACT, SPE, self.model_dir, from_saved=LOAD_RL_FROM_SAVED)
        self.episode_return, self.episode_length = 0, 0

        # active period detection tools
        self.APD = Active_Period_Detector(self.apd_queue, self.send_queue)
        self.config_flag = False

        # threads to handle streams
        sender = Thread(target=self.Send_AP, args=[])
        rewarder = Thread(target=self.process_rewards, args=[])
        receiver = Thread(target=self.receiver, args=[])
        receiver.setDaemon(True)
        sender.setDaemon(True)
        rewarder.setDaemon(True)
        receiver.start()
        sender.start()
        rewarder.start()

    def stream(self):
        start = time.time()
        while True:
            if not self.input.empty():
                chunk = channelize(self.input.get(), self.phaser, self.filt)
                chunk = chunk[10 * UPSAMPLE_FACTOR:-10 * UPSAMPLE_FACTOR]
                if not self.config_flag:
                    self.APD.configNF(chunk, 'low')
                    #self.APD.static_config(8.860483e-5, 9.70097, 9.50195514, 'low')
                    # self.APD.configNF(chunk, 'high')
                    self.APD.start_consuming()
                    self.config_flag = True
                self.apd_queue.put(chunk)
                start = time.time()
            else:
                time.sleep(0.2)
                if time.time() - start > 10 and SAVE_RL:
                    print("saving RL...")
                    self.my_ppo.save()
                    break

    def Send_AP(self):
        while True:
            if not self.send_queue.empty():
                while True:
                    if self.output.empty():
                        self.act_and_send(self.send_queue.get())
                        break
                    else:
                        time.sleep(0.05)
            else:
                time.sleep(0.05)

    # receiver waits for response packets from the server to train the RL on
    def receiver(self):
        while True:
            if not self.rew_queue.empty():
                a_rew = self.rew_queue.get()
                self.Rewards[a_rew[0]] = (a_rew[1])
            else:
                time.sleep(0.25)

    # Thread to process rewards IN ORDER, rewards processed out of order showed poor results
    def process_rewards(self):
        while True:
            if self.rew_counter in self.Rewards:
                print(f"Reward Counter: {self.rew_counter}")

                # get history to process
                hist = self.History.pop(self.rew_counter)
                num_dec = self.Rewards.pop(self.rew_counter)
                obs = hist.get_obs()
                obs = obs.reshape(1, -1)

                # keep tabs on false positive rates of APs
                if len(self.Prev_send_rates) > 100:
                    self.Prev_send_rates.pop(0)
                success_rates = get_prev_success(self.Prev_send_rates)
                self.Prev_send_rates.append(max(sum(num_dec), 0))


                # store observations internally to PPO agent
                reward = hist.calc_rewards(num_dec, success_rates)
                self.episode_return += reward
                self.episode_length += 1
                self.my_ppo.store_step(obs, hist.logits, hist.action, reward)
                store_vals = hist.get_string() + f"\t{reward}\t{sum(num_dec)}"
                store_results(self.save_dir + 'Single_rewards.txt', store_vals)

                # after each epoch, train the RL models
                if self.episode_length == SPE:
                    # store_results('Episode_rewards.txt', self.episode_return)
                    last_value = 0
                    self.my_ppo.critic(obs.reshape(1, -1))
                    self.my_ppo.buffer.finish_trajectory(last_value)
                    self.my_ppo.train_self()
                    self.episode_length = 0
                    self.episode_return = 0

                self.rew_counter += 1
            else:
                time.sleep(0.25)

    # Helper function to determine an action for a given active period, and convert it into a sendable form
    #   in:     sess is the session information
    #   out:    an active session object
    def act_and_send(self, Pkt_To_Send):
        #print(f"inq = {self.input.qsize()} outq = {self.output.qsize()}")
    
        info = Pkt_To_Send[0]
        buffer = Pkt_To_Send[1]
        start_t = Pkt_To_Send[2]

        # organize info
        mag = info[3]
        hist = info[4].tolist()
        dden = info[5]
        pkt_size = info[1] - info[0]
        pkt_size_norm = pkt_size / FS
        buff_size = self.send_queue.qsize() / QUEUE_SIZE

        # format and normalize bandwidth properties
        act_time = time.time()
        trim_by_time(act_time, self.Prev_Send, 10)

        bandwid = BW_sim()  # bandwidth in bps not Bps
        if bandwid == 0:
            bandwid = 1000000
        BW_obs = math.log(bandwid / 50000000) / 3

        BW_possible, BW_used = integrate_BW(self.Prev_Send, act_time, -1)
        BW_ratio = (BW_used * 8) / BW_possible
        BW_possible, BW_used = integrate_BW(self.Prev_Send, act_time, 5)
        last_BW = (((BW_used * 8) / BW_possible) / 5)
        BW_possible, BW_used = integrate_BW(self.Prev_Send, act_time, 10)
        last_BW2 = (((BW_used * 8) / BW_possible) / 5)

        # from session information (state) determine action
        observation = np.array([BW_obs, BW_ratio, last_BW, last_BW2, pkt_size_norm, buff_size, mag, dden] + hist,
                               dtype=np.float32)
        observation = observation.reshape(1, -1)
        logits, act = self.my_ppo.sample_action(observation)
        action = 0
        if ACT_COMPRESS:
            action = dden * act[0].numpy() * 5

        # create active session object
        zz = Active_Session(ID, self.channel, self.pkt_counter, act_time, action, buffer, info[2],
                            SEND_SIZE, FS)
        state = State(mag, hist, dden, BW_ratio, pkt_size, logits, act, 0, act_time, pkt_size_norm,
                      buff_size, last_BW, last_BW2, BW_obs)

        print(f"Send Packet: {self.pkt_counter}")
        self.History[self.pkt_counter] = state
        if self.pkt_counter - 1 in self.History:
            self.History[self.pkt_counter - 1].post_buff_size = buff_size

        if buffer is not None:
            pkt = zz.build_packet()
            self.History[self.pkt_counter].comp_size = len(pkt)
            self.History[self.pkt_counter].latency = min(time.time() - start_t, 20)
            if (len(pkt) * 8) / bandwid > ALPHA:	# todo: if ALPHA is devided by the number of channels, fairness is hacked but definintly not best
                print("packet actually dropped due to latency limitations")
                self.Rewards[self.pkt_counter] = ([-1, -1, -1, -1, -1, -1])
                self.Prev_Send.append([act_time, 0, bandwid])
            else:
                self.Prev_Send.append([act_time, state.comp_size, bandwid])
                self.output.put(pkt)
        else:
            print("packet actually dropped, buffer was too full")
            self.Rewards[self.pkt_counter] = ([-1, -1, -1, -1, -1, -1])
            self.Prev_Send.append([act_time, 0, bandwid])
            
        del buffer
        del zz
        self.pkt_counter += 1
