import os
from datagen import *
import mask
from models import unet
import logging
from misc import *
import argparse

CURR_DIR = os.getcwd()
DATA_PATH = os.path.join(CURR_DIR,'data')
LOG_PATH = os.path.join(CURR_DIR,'logs')





def main():
    #log_path = new_logdir(LOG_PATH)
    net = unet()
    print(net.summary())



if __name__ == '__main__':
    assert os.path.isdir(DATA_PATH), 'Missing data Directory'
    assert os.path.isdir(LOG_PATH), 'Missing logs directory'
    main()
