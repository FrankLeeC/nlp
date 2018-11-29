import collections

# word = ['a', 'b', 'c', 'c', 'd', 'a', 'a', 'g', 'h']
word = '12345234595'
count = [['UNK', -1]]
count.extend(collections.Counter(word).most_common(word.__len__()))
print('count:', count)
dic = dict()
for w, c in count:
    print(w, c, '======')
    dic[w] = len(dic)

print('===========================')
unk_count = 0
l = list()
for w in word:
    i = dic.get(w, 0)
    if i == 0:
        unk_count += 1
    l.append(i)
count[0][1] = unk_count
print('dic:', dic)
print('l:', l)
print('word:', word)
reversed_dict = dict(zip(dic.values(), dic.keys()))
print('reversed:', reversed_dict)


print('=======================================================')

# count.extend(collections.Counter(['侧堂堂挠堂堂']).most_common(self.data.__len__()))
#         for w, _ in count:
#             self.dictionary[w] = self.dictionary.__len__()
#         for w in self.data:
#             i = self.dictionary.get(w, 0)
#             if i != 0:
#                 self.data_set.append(i)
s = 'the cat sit on the mat'.split(' ')
count = []
count.extend(collections.Counter(s).most_common(6))
print(count)