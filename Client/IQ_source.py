import numpy as np
import os
import time
import socket


def file_load(channel_streams):
    # we need to wait for the streams to set up
    from client_config import FILE_PATH, RAW_FS, OVERLAP, THROTTLE_RATE
    direc = FILE_PATH
    files = os.listdir(direc)
    for file in files:
        fname = direc + file

        if file != '3_chan_test':
            continue
        print(f"Processing File: {file}")

        file_size = os.path.getsize(fname)
        start_time = time.time()
        offset = 0
        atime = time.time()
        while offset + RAW_FS * 8 < file_size - 100:
            f = open(fname)
            chunk = np.fromfile(f, dtype=np.complex64,
                                count=int(RAW_FS + OVERLAP * 2),
                                offset=int(offset))
            f.close()

            for chan in channel_streams:
                chan.put((chunk, atime))

            offset += RAW_FS * 8
            atime += 1
            time.sleep(THROTTLE_RATE - ((time.time() - start_time) % THROTTLE_RATE))

def UDP_load(channel_streams):
    from client_config import RAW_FS, OVERLAP
    t2 = '127.0.0.1'

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((t2, 2000))
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)

    recver = list()
    counter = 0
    tmp = np.array([])
    atime = time.time()

    while True:
        message, address = server_socket.recvfrom(2 ** 15)
        if len(message) > 0:
            #print(len(message))
            counter += len(message)
            recver.append(message)

            if counter > int(RAW_FS + OVERLAP * 2) * 8:
                final = np.array(recver).view(dtype=np.complex64)       # flatten
                final = np.concatenate((tmp, final))                    # combine with previous
                tmp = final[int(RAW_FS):]                               # get overlap
                final = final[0:int(RAW_FS + OVERLAP * 2)]              # final chunk
                for chan in channel_streams:
                    chan.put((final, atime))
                recver = list()
                counter = len(tmp) * 8
                atime += 1
        else:
            time.sleep(0.1)


def UDP_load2(chan, chan_num):
    from client_config import RAW_FS, OVERLAP
    t2 = '127.0.0.1'
    if chan_num == 1:
        port = 4900
    elif chan_num == 2:
        port = 5000
    elif chan_num == 3:
        port = 5100
    elif chan_num == 4:
        port = 5200
    elif chan_num == 0:
        port = 2000
    elif chan_num == 5:
        port = 2010
    elif chan_num == 6:
        port = 1998
    elif chan_num == 7:
        port = 2020
    elif chan_num == 8:
        port = 2030

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((t2, port))
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)

    recver = list()
    counter = 0
    tmp = np.array([])
    atime = time.time()

    while True:
        message, address = server_socket.recvfrom(2 ** 15)
        if len(message) > 0:
            #print(len(message))
            counter += len(message)
            recver.append(message)

            if counter > int(RAW_FS + OVERLAP * 2) * 8:
                final = np.array(recver).view(dtype=np.complex64)       # flatten
                final = np.concatenate((tmp, final))                    # combine with previous
                tmp = final[int(RAW_FS):]                               # get overlap
                final = final[0:int(RAW_FS + OVERLAP * 2)]              # final chunk
                chan.put((final, time.time()))
                recver = list()
                counter = len(tmp) * 8
                atime += 1
        else:
            time.sleep(0.1)
