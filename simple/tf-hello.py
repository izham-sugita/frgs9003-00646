from __future__ import print_function

import tensorflow as tf

hello = tf.constant("Hello, tensorflow!")

#start session
sess = tf.Session()

#run session
print(sess.run(hello))
