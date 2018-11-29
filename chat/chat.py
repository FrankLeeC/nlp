# -*- coding: utf-8 -*-

import numpy as np
import layer
import cell
import logging
import signal
import win_unicode_console
win_unicode_console.enable()

 
logger = logging.getLogger("stack_lstm")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler("train.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
# logger.removeHandler(file_handler)

use_log=False


def output(s):
    if not use_log:
        print(s)
    else:
        logger.info(s)

def terminate(signum, frame):
    output('terminate')
    t.save()
    word = t.test()
    output('last sample:' + word)
    exit(0)

signal.signal(signal.SIGTERM, terminate)


class Network:

    def __init__(self):
        self.hsize = 200
        self.embedding_size = 100
        self.vsize = 300
        self.layer_size = 3

    def create_layer(self):
        encoder_embedding = layer.WordEmbeddingLayer(shape=[self.embedding_size, self.vsize])
        for i in range(self.layer_size):
            is_output = (i == self.layer_size - 1)
            layer.EncoderLayer(5, cell.EncoderGRU(self.hsize, self.embedding_size), is_output=is_output)

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

    


t = Network()
if __name__ == '__main__':
    try:
        for i in range(10000):
            t.train()
    except KeyboardInterrupt as e:
        output('stop!')
    finally:
        output('over!')
        t.save()
        word = t.test()
        outpput(word)