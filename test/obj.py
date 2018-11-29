# -*- coding:utf-8 -*-

class Test():
    
    name = 7

    def __init__(self):
        pass
    

    def init(self):
        self.name += 1


if __name__ == '__main__':
    a = Test()
    print(a.name)
    b = Test()
    b.init()
    print(a.name)
    print(b.name)    
    print(Test.name)