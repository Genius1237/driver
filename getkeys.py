# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check(keys="WSAD"):
    keys_d = {}
    for key in keys:
        if wapi.GetAsyncKeyState(ord(key)):
            keys_d[key]=True
        
    #print(keys_d)
    return keys_d