"""
 Automatically start redis server, add worker nodes to ray's redis server and sync code to worker machines

 __author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import ray
import socket
import os
import time
from genome import CPPNGenome
import csv


def initialise_hpc(worker_list):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    try:
        ray.init(address=ip+":54504")
    except:
        os.system("gnome-terminal -e 'bash -c \"ray start --head --redis-port=54504; exec bash\"'")
        time.sleep(2)
        ray.init(address=ip + ":54504")
    if worker_list is not None:
        worker_ips = []
        with open(worker_list+".csv") as csvfile:
            rows = csv.reader(csvfile, delimiter=",")
            next(rows, None)
            for row in rows:
                worker_ips.append(row[0])
        for ip in worker_ips:
            setup_worker(ip)
    ray.register_custom_serializer(CPPNGenome, use_pickle=True)


def setup_worker(ip):
    # ssh to worker
    # pull latest code
    # connect worker to redis server
    pass