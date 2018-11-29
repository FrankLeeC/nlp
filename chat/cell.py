# -*- coding: utf-8 -*-

import numpy as np
import optimizer as opt
import param as pm
import function as fn

# EncoderGRU CELL -----------------------------------------------------------------------------
class EncoderGRU:

    def __init__(self, hsize, isize):
        self.isize = isize
        self.param = pm.EncoderParam(hsize, isize)
        self.init_optimizer()
        self.clear()

    def init_optimizer(self):
        self.wr_g = opt.create_optimizer()
        self.ur_g = opt.create_optimizer()
        self.br_g = opt.create_optimizer()
        self.wz_g = opt.create_optimizer()
        self.uz_g = opt.create_optimizer()
        self.bz_g = opt.create_optimizer()
        self.whs_g = opt.create_optimizer()
        self.uhs_g = opt.create_optimizer()
        self.bhs_g = opt.create_optimizer()

    def clear(self):
        self.r_arr, self.z_arr, self.hs_arr, self.h_arr, self.x_arr = [], [], [], [], []
        
    def clear_grad(self):
        self.dwr = np.zeros_like(self.param.wr)
        self.dur = np.zeros_like(self.param.ur)
        self.dbr = np.zeros_like(self.param.br)
        self.dwz = np.zeros_like(self.param.wz)
        self.duz = np.zeros_like(self.param.uz)
        self.dbz = np.zeros_like(self.param.bz)
        self.dwhs = np.zeros_like(self.param.whs)
        self.duhs = np.zeros_like(self.param.uhs)
        self.dbhs = np.zeros_like(self.param.bhs)

    def step(self, x, h_prev):
        r = fn.sigmoid(np.dot(self.param.wr, x) + self.param.ur*h_prev + self.param.br)
        z = fn.sigmoid(np.dot(self.param.wz, x) + self.param.uz*h_prev + self.param.bz)
        hs = fn.tanh(np.dot(self.param.whs, x) + self.param.uhs*(r*h_prev) + self.param.bhs)
        h = z*h_prev + (1-z)*hs
        return r, z, hs, h

    def store(self, x, r, z, hs, h):
        self.x_arr.append(x)                
        self.r_arr.append(r)
        self.z_arr.append(z)
        self.hs_arr.append(hs)
        self.h_arr.append(h)

    def backward(self, err, dh_next, clip = 1.0):
        l = err.__len__()
        dx_arr = [np.zeros((self.isize, 1))] * err.__len__()
        h_prev = np.zeros((self.param.hsize, 1))
        for i in range(l):
            j = l - i - 1
            if j >= 1:
                h_prev = self.h_arr[j-1]
            dh = err[j] + dh_next
            dz = dh*(self.h_arr[j-1]-self.hs_arr[j])
            dhs = dh*(1-self.z_arr[j])
            self.duhs += dhs*fn.dtanh(self.hs_arr[j])*(self.r_arr[j]*h_prev)
            self.dwhs += np.dot(dhs*fn.dtanh(self.hs_arr[j]), self.x_arr[j].T)
            self.dbhs += dhs*fn.dtanh(self.hs_arr[j])
            self.duz += dz*fn.dsigmoid(self.z_arr[j])*h_prev
            self.dwz += np.dot(dz*fn.dsigmoid(self.z_arr[j]), self.x_arr[j].T)
            self.dbz += dz*fn.dsigmoid(self.z_arr[j])
            dr = dhs*fn.dtanh(self.hs_arr[j])*self.param.uhs*h_prev
            self.dur += dr*fn.dsigmoid(self.r_arr[j])*h_prev
            self.dwr += np.dot(dr*fn.dsigmoid(self.r_arr[j]), self.x_arr[j].T)
            self.dbr += dr*fn.dsigmoid(self.r_arr[j])
            dh_next = dh*(self.z_arr[j]) + dhs*fn.dtanh(self.hs_arr[j])*self.param.uhs*self.r_arr[j] + dz*fn.dsigmoid(self.z_arr[j])*self.param.uz + dr*fn.dsigmoid(self.r_arr[j])*self.param.ur
            dx = np.dot(self.param.whs.T, dhs*fn.dtanh(self.hs_arr[j])) + np.dot(self.param.wz.T, dz*fn.dsigmoid(self.z_arr[j])) + np.dot(self.param.wr.T, dr*fn.dsigmoid(self.r_arr[j]))
            dx_arr[j] = dx
        self.clear()
        return dx_arr

    def update(self, batch_size, clip=0.5):
        self.param.whs += self.whs_g.get_grad(np.clip(self.dwhs/batch_size, -clip, clip))
        self.param.uhs += self.uhs_g.get_grad(np.clip(self.duhs/batch_size, -clip, clip))
        self.param.bhs += self.bhs_g.get_grad(np.clip(self.dbhs/batch_size, -clip, clip))
        self.param.wz += self.wz_g.get_grad(np.clip(self.dwz/batch_size, -clip, clip))
        self.param.uz += self.uz_g.get_grad(np.clip(self.duz/batch_size, -clip, clip))
        self.param.bz += self.bz_g.get_grad(np.clip(self.dbz/batch_size, -clip, clip))
        self.param.wr += self.wr_g.get_grad(np.clip(self.dwr/batch_size, -clip, clip))
        self.param.ur += self.ur_g.get_grad(np.clip(self.dur/batch_size, -clip, clip))
        self.param.br += self.br_g.get_grad(np.clip(self.dbr/batch_size, -clip, clip))
        self.clear_grad()

# End of EncoderGRU CELL -----------------------------------------------------------------------------





# DecoderGRU CELL -----------------------------------------------------------------------------
class DecoderGRU:

    def __init__(self, hsize, isize, csize, c):
        self.isize = isize
        self.c = c
        self.param = pm.DecoderParam(hsize, isize, csize)
        self.init_optimizer()
        self.clear()

    def init_optimizer(self):
        self.wr_g = opt.create_optimizer()
        self.ur_g = opt.create_optimizer()
        self.cr_g = opt.create_optimizer()
        self.wz_g = opt.create_optimizer()
        self.uz_g = opt.create_optimizer()
        self.cz_g = opt.create_optimizer()
        self.whs_g = opt.create_optimizer()
        self.uhs_g = opt.create_optimizer()
        self.chs_g = opt.create_optimizer()

    def clear(self):
        self.r_arr, self.z_arr, self.hs_arr, self.h_arr, self.x_arr = [], [], [], [], []
        
    def clear_grad(self):
        self.dwr = np.zeros_like(self.param.wr)
        self.dur = np.zeros_like(self.param.ur)
        self.dcr = np.zeros_like(self.param.cr)
        self.dwz = np.zeros_like(self.param.wz)
        self.duz = np.zeros_like(self.param.uz)
        self.dcz = np.zeros_like(self.param.cz)
        self.dwhs = np.zeros_like(self.param.whs)
        self.duhs = np.zeros_like(self.param.uhs)
        self.dchs = np.zeros_like(self.param.chs)

    def step(self, x, h_prev):
        r = fn.sigmoid(np.dot(self.param.wr, x) + self.param.ur*h_prev + np.dot(self.param.cr, self.c))
        z = fn.sigmoid(np.dot(self.param.wz, x) + self.param.uz*h_prev + np.dot(self.param.cz, self.c))
        hs = fn.tanh(np.dot(self.param.whs, x) + r*(self.param.uhs*h_prev + np.dot(self.param.chs, self.c)))
        h = z*h_prev + (1-z)*hs
        return r, z, hs, h

    def store(self, x, r, z, hs, h):
        self.x_arr.append(x)                
        self.r_arr.append(r)
        self.z_arr.append(z)
        self.hs_arr.append(hs)
        self.h_arr.append(h)

    def backward(self, err, dh_next, clip = 1.0):
        l = err.__len__()
        dx_arr = [np.zeros((self.isize, 1))] * err.__len__()
        h_prev = np.zeros((self.param.hsize, 1))
        for i in range(l):
            j = l - i - 1
            if j >= 1:
                h_prev = self.h_arr[j-1]
            dh = err[j] + dh_next
            dz = dh*(h_prev- self.hs_arr[j])
            dhs = dh*(1-self.z_arr[j])
            self.duhs += dhs*fn.dtanh(self.hs_arr[j])*self.r_arr[j]*h_prev
            self.dwhs += dhs*fn.dtanh(self.hs_arr[j])*self.x_arr[j]
            self.dchs += np.dot(np.dot(dhs*fn.dtanh(self.hs_arr[j]))*self.r_arr[j], self.c.T)
            self.duz += dz*fn.dsigmoid(self.z_arr[j])*h_prev
            self.dwz += np.dot(dz*fn.dsigmoid(self.z_arr[j]), self.x_arr[j].T)
            self.dcz += np.dot(dz*fn.dsigmoid(self.z_arr[j]), self.c.T)
            dr = dhs*fn.dtanh(self.hs_arr[j])*(self.param.uhs*h_prev+np.dot(self.param.chs, self.c))
            self.dur += dr*fn.dsigmoid(self.r_arr[j])*h_prev
            self.dwr += np.dot(dr*fn.dsigmoid(self.r_arr[j]), self.x_arr[j].T)
            self.dcr += np.dot(dr*fn.dsigmoid(self.r_arr[j]), self.c.T)
            dh_next = dh*self.z_arr[j] + dhs*fn.dtanh(self.hs_arr[j])*self.r_arr[j]*self.param.uhs + dz*fn.dsigmoid(self.z_arr[j])*self.param.uz + dr*fn.dsigmoid(self.r_arr[j])*self.param.ur
            dx = np.dot(self.param.whs.T, dhs*fn.dtanh(self.hs_arr[j])) + np.dot(self.param.wz.T, dz*fn.dsigmoid(self.z_arr[j])) + np.dot(self.param.wr.T, dr*fn.dsigmoid(self.r_arr[j]))
            dx_arr[j] = dx
        self.clear()
        return dx_arr

    def update(self, batch_size, clip=0.5):
        self.param.whs += self.whs_g.get_grad(np.clip(self.dwhs/batch_size, -clip, clip))
        self.param.uhs += self.uhs_g.get_grad(np.clip(self.duhs/batch_size, -clip, clip))
        self.param.chs += self.chs_g.get_grad(np.clip(self.dchs/batch_size, -clip, clip))
        self.param.wz += self.wz_g.get_grad(np.clip(self.dwz/batch_size, -clip, clip))
        self.param.uz += self.uz_g.get_grad(np.clip(self.duz/batch_size, -clip, clip))
        self.param.cz += self.cz_g.get_grad(np.clip(self.dcz/batch_size, -clip, clip))
        self.param.wr += self.wr_g.get_grad(np.clip(self.dwr/batch_size, -clip, clip))
        self.param.ur += self.ur_g.get_grad(np.clip(self.dur/batch_size, -clip, clip))
        self.param.cr += self.cr_g.get_grad(np.clip(self.dcr/batch_size, -clip, clip))
        self.clear_grad()

# End of DecoderGRU CELL -----------------------------------------------------------------------------