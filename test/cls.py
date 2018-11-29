# -*- coding: utf-8 -*-

a = 13

class O:
    
    def doo(self, b):
        b += 1
        return b

o = O()

class T:

    def do(self):
        global a
        a = o.doo(a)

if __name__ == '__main__':
    t = T()
    t.do()
    t.do()
    t.do()
    print(a)    