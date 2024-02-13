import socket
import time
import sys

ClientMultiSocket = socket.socket()
host = sys.argv[1]
port = 2004

ch_id = sys.argv[2]
bs_id = sys.argv[3]

print('Waiting for connection response')
try:
    ClientMultiSocket.connect((host, port))
except socket.error as e:
    print(str(e))
    
    
config = str(ch_id) + str(bs_id)
res = ClientMultiSocket.send(str.encode(config))
time.sleep(3)

cnt = int(sys.argv[2])

while True:
    Input = 'count_num: ' + str(cnt)
    ClientMultiSocket.send(str.encode(Input))
    res = ClientMultiSocket.recv(1024)
    print(res.decode('utf-8'))
    time.sleep(3)
    cnt += 1
ClientMultiSocket.close()
