import sys
from load_model import train
import time
import termios
import os
import select

def GetChar(Block=True):
  if Block or select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
    return sys.stdin.read(1)
  return None
while True:
    start = time.time()
    end = time.time()
    key = None
    print('Type E to exit or else it will train one epoch in 60 seconds')
    while ((end-start)< 60) & (key == None):
        key = GetChar(False)
        if key == 'E':
            exit()
        end = time.time()
    train()