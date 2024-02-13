import socket
import time
import sys

ClientMultiSocket = socket.socket()
host = '127.0.0.1'
port = 65432

print('Waiting for connection response')
try:
    ClientMultiSocket.connect((host, port))
except socket.error as e:
    print(str(e))
    
time.sleep(1000)
    
#msg_len = 12
#message = msg_len.to_bytes(4, 'big'), ch_id.to_bytes(4, 'big')
#    
#config = str(ch_id) + str(bs_id)
#res = ClientMultiSocket.send(str.encode(config))
#time.sleep(3)
#
#cnt = int(sys.argv[2])
#
#while True:
#    Input = 'count_num: ' + str(cnt)
#    ClientMultiSocket.send(str.encode(Input))
#    res = ClientMultiSocket.recv(1024)
#    print(res.decode('utf-8'))
#    time.sleep(3)
#    cnt += 1
#ClientMultiSocket.close()
