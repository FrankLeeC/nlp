# -*- coding=utf-8 -*-

a = 10

def p():
    global a
    a += 20
    print(a)

# global a
if __name__ == '__main__':
    a -= a * 0.01
    p()