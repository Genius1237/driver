from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,S,A,D
import random
import time
import cv2
import pickle
from multiprocessing import Process, Pipe

class ParallelWriter():
    
    @staticmethod
    def worker(conn):
        while(True):
            data = conn.recv()
            pickle.dump(data,open('{}.pickle'.format(time.time()),'wb'))
    
    def __init__(self,limit):
        self.local,self.remote = Pipe()
        self.process = Process(target=ParallelWriter.worker, args=(self.remote,))
        self.process.start()
        self.remote.close()
        self.list = []
        self.len = 0
        self.limit = limit
    
    def append(self,item):
        self.list.append(item)
        self.len+=1

        if self.len == self.limit:
            self.local.send(self.list)
            self.list = []
            self.len = 0

if __name__ == "__main__":  
    '''
    #input()
    while(True):
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
    '''
    p = ParallelWriter(limit=100)
    t = time.time()

    while(True):
        #image = grab_screen(window_title='Grand Theft AUto V',region=(1,31,1067,630))
        image = grab_screen(region=(1,31,1067,630))
        keys = key_check()
        image = cv2.resize(image,(640,360))
        p.append((image,keys))
        #pickle.dump((image,keys),open('{}.pickle'.format(t),'wb'))
        #data.append((image,keys))
        t_new = time.time()
        print((t_new-t)*1000)
        t = t_new
    