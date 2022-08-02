#!/usr/bin/env python3

import socket

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP

# Enable port reusage so we will be able to run multiple clients and servers on single (host, port).
# Do not use socket.SO_REUSEADDR except you using linux(kernel<3.9): goto https://stackoverflow.com/questions/143$
# For linux hosts all sockets that want to share the same address and port combination must belong to processes t$
# So, on linux(kernel>=3.9) you have to run multiple servers and clients under one user to share the same (host, $
# Thanks to @stevenreddie
client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

# Enable broadcasting mode
client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# app is keep sending udp broardcast into port 5005
client.bind(("", 5005))
while True:
    # Thanks @seym45 for a fix
    data, addr = client.recvfrom(1024)
    print("recv: %s" % data)