# -*- coding: utf-8 -*-
import numpy as np

class EncoderParam:

    def __init__(self, hsize, isize):
        self.hsize = hsize
        self.isize = isize
        d = isize
        self.wr = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.ur = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.br = (2/d)*np.random.rand(hsize, 1) - 1/d
        self.wz = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.uz = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.bz = (2/d)*np.random.rand(hsize, 1) - 1/d
        self.whs = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.uhs = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.bhs = (2/d)*np.random.rand(hsize, 1) - 1/d

    def save(self, index):
        np.save('./save/encoder_'+str(index)+'_wr', self.wr)
        np.save('./save/encoder_'+str(index)+'_ur', self.ur)
        np.save('./save/encoder_'+str(index)+'_br', self.br)
        np.save('./save/encoder_'+str(index)+'_wz', self.wz)
        np.save('./save/encoder_'+str(index)+'_uz', self.uz)
        np.save('./save/encoder_'+str(index)+'_bz', self.bz)
        np.save('./save/encoder_'+str(index)+'_whs', self.whs)
        np.save('./save/encoder_'+str(index)+'_uhs', self.uhs)        
        np.save('./save/encoder_'+str(index)+'_bhs', self.bhs)


class DecoderParam:

    def __init__(self, hsize, isize, csize):
        self.hsize = hsize
        self.isize = isize
        self.csize = csize
        d = isize
        self.whs = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.uhs = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.chs = (2/d)*np.random.rand(hsize, csize) - 1/d
        self.wz = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.uz = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.cz = (2/d)*np.random.rand(hsize, csize) - 1/d
        self.wr = (2/d)*np.random.rand(hsize, isize) - 1/d
        self.ur = (2/d)*np.random.rand(hsize, hsize) - 1/d
        self.cr = (2/d)*np.random.rand(hsize, csize) - 1/d

    def save(self, index):
        np.save('./save/decoder_'+str(index)+'_wr', self.wr)
        np.save('./save/decoder_'+str(index)+'_ur', self.ur)
        np.save('./save/decoder_'+str(index)+'_cr', self.cr)
        np.save('./save/decoder_'+str(index)+'_wz', self.wz)
        np.save('./save/decoder_'+str(index)+'_uz', self.uz)
        np.save('./save/decoder_'+str(index)+'_cz', self.cz)
        np.save('./save/decoder_'+str(index)+'_whs', self.whs)
        np.save('./save/decoder_'+str(index)+'_uhs', self.uhs)        
        np.save('./save/decoder_'+str(index)+'_chs', self.chs)
