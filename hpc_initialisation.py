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


def initialise_hpc(worker_list, local_mode=False, log_to_driver=True):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    # Restart master node
    os.system("ray stop")
    import multiprocessing
    logical_processors = multiprocessing.cpu_count()-1
    redis_port = "15357"
    address = ip + ":" + redis_port
    os.system("ray start --head --redis-port="+redis_port+" --include-webui --load-code-from-local")  # start server
    if worker_list is not None:
        host_address = address
        # Build modules for sending to worker nodes
        path = os.path.dirname(os.path.realpath(__file__))
        os.system("cd "+path+" | python setup.py sdist")
        worker_ips = []
        with open(worker_list+".csv") as csvfile:
            rows = csv.reader(csvfile, delimiter=",")
            next(rows, None)
            for row in rows:
                worker_ips.append({"ip": row[0], "user": row[1], "pw": row[2]})
        path += "/dist/"
        for file in os.listdir(path):
            if "NAS" in file:
                pkg_name = file
                break
        for w in worker_ips:
            setup_worker(w["ip"], w["user"], w["pw"], path, pkg_name, host_address)
    ray.init(address=address, local_mode=local_mode, log_to_driver=log_to_driver)  # initialise cluster for calling remote on master node
    if not local_mode:
        ray.register_custom_serializer(CPPNGenome, use_pickle=True)


def setup_worker(ip, user, pw, path_to_pkg, pkg_name, host_address):
    # ssh to worker
    # pull latest code
    # connect worker to redis server
    host = user+"@"+ip
    os.system("sshpass -p \""+pw+"\" scp -o StrictHostKeyChecking=no "+path_to_pkg+pkg_name+" "+host+":/tmp")
    os.system("sshpass -p \""+pw+"\" ssh -o StrictHostKeyChecking=no "+host+" -t \"bash -l -c -i 'pip install /tmp/NAS-0.1.tar.gz'\"")
    os.system("sshpass -p \""+pw+"\" ssh -o StrictHostKeyChecking=no "+host+" -t \"bash -l -c -i 'ray stop'\"")
    #py_cmds = "bash -l -c -i \"python -O -c 'import ray;ray.init(address=\""+host_address+"\",load_code_from_local=True);'\""
    os.system("sshpass -p \""+pw+"\" ssh -o StrictHostKeyChecking=no "+host+" -t \"bash -l -c -i 'ray start --address="+host_address+" --load-code-from-local'\"")
    #os.system("sshpass -p \"" + pw + "\" ssh -o StrictHostKeyChecking=no " + host + " -t "+py_cmds)
