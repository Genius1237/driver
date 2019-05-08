import pickle
import os
from multiprocessing import Process, Pipe
import torch
import h5py

class ParallelWriter():
    
    @staticmethod
    def worker(path,conn):
        index=0
        while(True):
            data = conn.recv()
            file_name = os.path.join(path,'{}.pickle'.format(index))
            with open(file_name,'wb') as f:
                pickle.dump(data,f)
            index+=1
    
    def __init__(self,path,limit=64):
        self.path = path
        self.local,self.remote = Pipe()
        self.process = Process(target=ParallelWriter.worker, args=(self.path,self.remote,))
        self.process.start()
        self.remote.close()
        self.list = []
        self.len = 0
        self.limit = limit
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
    
    def append(self,item):
        self.list.append(item)
        self.len+=1

        if self.len == self.limit:
            self.local.send(self.list)
            self.list = []
            self.len = 0

    def __del__(self):
        #self.local.send(self.list)
        self.process.terminate()

class ParallelWriterHDF5():
    def __init__(self,path):
        self.path = path
        self.created = False
    
    def append(self,item):
        X,Y = item
        
        if not self.created:
            with h5py.File(self.path,"w") as file:
                x = file.create_dataset('X',shape=(1,)+X.shape,maxshape=((None,)+X.shape))
                y = file.create_dataset('Y',shape=(1,)+Y.shape,maxshape=((None,)+Y.shape),dtype='i4')

                x[-1] = X
                y[-1] = Y

            self.created = True
        
        else:
            with h5py.File(self.path,"a") as file:
                x = file['X']
                y = file['Y']

                x.resize(x.shape[0]+1,axis=0)
                y.resize(y.shape[0]+1,axis=0)

                x[-1] = X
                y[-1] = Y

class DataReader():

    def __init__(self,path,file_name):
        self.path = path
        self.n_files = len(os.listdir(path))
        self.index = 0
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < self.n_files:
            file = os.path.join(self.path,'{}.pickle'.format(self.index))
            with open(file,'rb') as f:
                data = pickle.load(f)
            X = torch.Tensor([d[0] for d in data])
            Y = torch.Tensor([d[0] for d in data])
            return X,Y
        else:
            raise StopIteration


def preprocess_raw(path):
    
    raw_data = []
    for file in os.listdir(path):
        if file != 'final.pickle':
            f = os.path.join(path,file)
            with open(f,'rb') as pf:
                raw_data.extend(pickle.load(pf))
    
    data_X = []
    data_Y = []
    for d in raw_data:
        frame = d[0]
        action_raw = d[1]
        
        """
        0 - Forward
        1 - Forward Right
        2 - Back Right
        3 - Back
        4 - Back Left
        5 - Forward Left
        """
        if 'W' in action_raw:
            if 'D' in action_raw:
                action = 1
            elif 'A' in action_raw:
                action = 5            
            else:              
                action = 0 
            data.append((frame,action))
        elif 'S' in action_raw:
            if 'D' in action_raw:
                action = 2
            elif 'A' in action_raw:
                action = 4
            else:     
                action = 3
            data.append((frame,action))

    save_path = os.path.join(path,'final.pickle')
    with open(f,'wb') as pf:
        pickle.dump(pf)