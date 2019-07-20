import numpy as np
import sys
import tensorflow as tf

#a = np.ndarray(shape=(3,3), dtype=tf.float16) #wrong! numpy don't get float16

#eager execution
#tf.compat.v1.enable_eager_execution()


x = tf.constant([1.25, 5.45], dtype=tf.float16)

a = tf.constant(np.arange(1.0,10.0, dtype=np.float32),
                shape=[3,3])

#cast to float16
a=tf.cast(a, tf.float16)
#print(a)

tf.print(a, output_stream=sys.stdout)


c = tf.matmul(a,a)

print(c)

sess = tf.compat.v1.Session()
with sess.as_default():
    tensor = a
    print_op = tf.print("tensors:\n", a,
                        output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
      multiplication = tf.matmul(a,a)
    sess.run(multiplication)


