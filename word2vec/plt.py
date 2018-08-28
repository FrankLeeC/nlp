# -*- coding:utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['ZHCN']
font = matplotlib.font_manager.FontProperties(fname='F:/Programs/Python/Python35/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttc')

fig, ax = plt.subplots()
a = np.random.random(10)
ax.plot(a[0:5], a[5:], 'o')
plt.xlabel('一二三', fontproperties=font)
# ax.set_xlabel(u'一二三')
ax.text(a[0], a[6], u'abc', family='msyh', ha='right', wrap=True)
plt.show(block=True)