from grabscreen import grab_screen
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W,S,A,D
import random
import time
import cv2
import utils

if __name__ == "__main__":  
    '''
    #input()
    while(True):
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
    '''
    p = utils.ParallelWriter('temp',limit=64)
    t = time.time()

    try:
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
    except KeyboardInterrupt:
        del p    