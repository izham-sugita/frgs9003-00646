import tensorflow as tf
import sys
import numpy as np

#eager execution allow the printing of elements
tf.compat.v1.enable_eager_execution()

#tensor = tf.range(10)
tensor = tf.constant(np.arange(1.0,10.0,dtype=np.float32),
                     shape=[3,3])

a = tf.cast(tensor, tf.float16)

#tf.print("tensors:\n", tensor, output_stream=sys.stdout)
#print(tensor)

c = tf.matmul(a,a) #add operation
print(c)

#compare to numpy
b = np.arange(1.0,10.0,dtype=np.float32).reshape((3,3))
d = np.matmul(b,b)
print(d)
