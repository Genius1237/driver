from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,S,A,D
import random
import time
import cv2
import utils
import numpy as np
import argparse
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",type=str,default='temp.hdf5',help='location of file to save training data to. MUST be a .hdf5 file')
    parser.add_argument("--window-resolution",type=int,nargs=2,default=[1067,600],help='resolution of the game window. 2 integers')
    parser.add_argument("--titlebar-adjustment",type=int,default=31,help='number of pixels taken up by the titlebar, which is used to specify the region of the screen to capture')
    parser.add_argument("--save-test-image",action="store_true",help='saves the capture as output.png . Useful for finetuning window capture area')

    args = parser.parse_args()

    try:
        assert args.file[-5:] == '.hdf5'
    except:
        print("File type MUST be hdf5")
        exit()

    file = args.file
    titlebar_adjustment = args.titlebar_adjustment
    width = args.window_resolution[0]
    height = args.window_resolution[1]


    start = False
    p = utils.ParallelWriterHDF5(file)
    t = time.time()

    while(True):
        if not start:
            if "X" in key_check("X"):
                start=True
                print("Starting Capture")
            continue
        
        image = grab_screen(window_title='Grand Theft Auto V',region=(1,1+titlebar_adjustment,width,height+titlebar_adjustment))
        #image = grab_screen(region=(1,31,1067,630))
        keys_t = key_check("WSADY")
        if "Y" in keys_t:
            print("Stopping Capture")
            break
        image = cv2.resize(image,(640,360))
        if args.save_test_image:
            cv2.imwrite('output.png',image)
            break
        keys = [0,0]
        if 'W' in keys_t:
            keys[0]=0
        else:
            keys[0]=1
        if 'D' in keys_t:
            keys[1]=0
        elif 'A' in keys_t:
            keys[1]=1
        else:
            keys[1]=2
        
        p.append((image,np.array(keys)))
        t_new = time.time()
        t = t_new