import tensorflow as tf
import numpy as np

def split():
    with tf.Session() as sess:
        a = [[1, 2, 3], [4, 5, 6]]
        x = tf.split(value=a, num_or_size_splits=3, axis=1)  # value: array, num_or_size_split: split size  axis: dimension
        c = sess.run(x)
        for i in c:
            print(i)

def unstack():
    with tf.Session() as sess:
        a = [[1, 2, 3], [4, 5, 6]]
        x = tf.unstack(a, axis=1)
        c = sess.run(x)
        print(c)  # [array([1, 4]), array([2, 5]), array([3, 6])]

def reshape():
    with tf.Session() as sess:
        a = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        a = np.array(a)
        b = tf.reshape(a, (2, -1, 5))
        print(sess.run(b))

def range_producer():
    with tf.Session() as sess:
        print(sess.run(tf.train.range_input_producer(10, shuffle=False).dequeue()))

def lookup():
    with tf.Session() as sess:
        # a = tf.constant([1, 2, 3, 4, 5])
        # b = tf.constant([6, 7, 8, 9, 10])
        # c = tf.constant([[1, 2], [0, 3]])
        # d = tf.nn.embedding_lookup([a, b], c)

        a = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], shape=(2, 5))
        c = tf.constant([[0, 1, 1], [1, 0, 0]])        
        d = tf.nn.embedding_lookup(a, c)   
        
             
        print(sess.run(d))


def ll():
    with tf.Session() as sess:
        a = tf.random_uniform([1, 2, 3, 4])
        b = tf.unstack(a, 1, axis=0)
        print(sess.run(a))
        print('----------')
        print(sess.run(b))


if __name__ == '__main__':
    # split()
    # unstack()
    # reshape()
    # range_producer()
    # lookup()
    ll()

"""
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
                                                               [4, 4, 4]]]
tf.strided_slice(input, [1, -1, 0], [2, -3, 3], [1, -1, 1]) ==>[[[4, 4, 4],
                                                                 [3, 3, 3]]]

"""                                                                 