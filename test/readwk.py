# -*- coding:utf-8 -*-

import codecs

with codecs.open('F:/Programs/Machine Learning/NLP/GoogleNewsVectors/GoogleNewsVectors.bin', mode='r', encoding='utf-8') as file:
    for i in range(1):
        line = file.readline()
        print(line)