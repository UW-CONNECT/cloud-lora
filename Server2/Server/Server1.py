import socket
import os
from _thread import *
ServerSideSocket = socket.socket()
host = socket.gethostbyname(socket.gethostname())
port = 2004
ThreadCount = 0

route_map = dict()

try:
    ServerSideSocket.bind((host, port))
except socket.error as e:
    print(str(e))
print('Socket is listening..')
ServerSideSocket.listen(5)

def multi_threaded_client(connection):
    while True:
        data = connection.recv(2048)
        response = 'Server message: ' + data.decode('utf-8')
        if not data:
            break
        connection.sendall(str.encode(response))
    connection.close()

while True:
    Client, address = ServerSideSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    config = Client.recv(2048).decode('utf-8')
    print('Worker Port: ' + str(address[1]))
    print('Client id: ' + str(config[0]))
    print('Client chan: ' + str(config[1]))
    
    start_new_thread(multi_threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSideSocket.close()
