# encoding=utf-8
import jieba

seg_list = jieba.cut("南京邮电大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式