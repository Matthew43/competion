import os
import tensorflow as tf

def evaluate():


    v = tf.one_hot([1,1,1,3,0,0,0],depth=4)

    with tf.Session() as sess:
        b = sess.run(v)

        d  = tf.argmax(b,axis=1)

        print(sess.run(d))
        print(v)
        print(b)


    pass



if __name__ == '__main__':
    evaluate()