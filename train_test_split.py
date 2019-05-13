import h5py
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='temp.hdf5',help='file to split into test and train')
parser.add_argument("--ratio",type=float,default=0.8,help='ratio of train:test')
parser.add_argument("--output-file",type=str,default='temp_processed.hdf5',help='file to store split dataset to')

args = parser.parse_args()

try:
    assert args.file[-5:] == '.hdf5'
except:
    print("File type MUST be hdf5")
    exit()

file = h5py.File(args.file,'r')

n_examples = file['Y'].shape[0]

with h5py.File(args.output_file,'w') as new_file:
    train_X = new_file.create_dataset('train_X',(1,)+file['X'].shape[1:],maxshape=(None,)+file['X'].shape[1:])
    train_Y = new_file.create_dataset('train_Y',(1,)+file['Y'].shape[1:],maxshape=(None,)+file['Y'].shape[1:])

    test_X = new_file.create_dataset('test_X',(1,)+file['X'].shape[1:],maxshape=(None,)+file['X'].shape[1:])
    test_Y = new_file.create_dataset('test_Y',(1,)+file['Y'].shape[1:],maxshape=(None,)+file['Y'].shape[1:])

    # x/(x+1) = r
    # x = xr + r
    # x(1-r) = r
    # x = r/(1-r)
    n = int(args.ratio/(1-args.ratio)) + 1
    
    trdone=[False]
    tedone=[False]
    for i in tqdm(range(n_examples)):
        if random.randint(1,n) == 1:
            #test
            target_X = test_X
            target_Y = test_Y
            bdone = tedone
        else:
            #train
            target_X = train_X
            target_Y = train_Y
            bdone = trdone

        if bdone[0]:
            target_X.resize(target_X.shape[0]+1,axis=0)
            target_Y.resize(target_Y.shape[0]+1,axis=0)

        target_X[-1] = file['X'][i]
        target_Y[-1] = file['Y'][i]
        bdone[0] = True