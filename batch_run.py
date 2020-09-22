# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 10:02 上午
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : batch_run



import threading
from threading import BoundedSemaphore, Lock
import os
import argparse
import sys

# cat cmd | python batch_run.py --gpus 0,1,2,3,0
'''python batch_run.py --gpus 0,0,1,1'''

parser = argparse.ArgumentParser(description="get args")
parser.add_argument('--gpus', action='store', dest='gpus', required=True)
args = parser.parse_args()

gpu_ids = [int(o) for o in args.gpus.split(',')]

sema = BoundedSemaphore(len(gpu_ids))
lst_lock = Lock()


class myThread(threading.Thread):
    def __init__(self, command):
        threading.Thread.__init__(self)
        self.command = command

    def run(self):
        sema.acquire()

        lst_lock.acquire()
        id_gpu = gpu_ids.pop()
        lst_lock.release()

        runCommand(self.command, id_gpu)

        lst_lock.acquire()
        gpu_ids.append(id_gpu)
        lst_lock.release()

        sema.release()


def runCommand(command, id_gpu):
    cmd = 'CUDA_VISIBLE_DEVICES=%d ' % (id_gpu) + command
    return os.system(cmd)


collect_thread = []

for line in sys.stdin:
    cmd = line.strip()
    thread = myThread(cmd)
    thread.start()
    collect_thread.append(thread)

for td in collect_thread:
    td.join()
