import tensorflow as tf
import sys
import numpy as np
import os
import time

#remove verbrose messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#eager execution allow the printing of elements
tf.compat.v1.enable_eager_execution()

#tensor = tf.range(10)
#tensor = tf.constant(np.arange(1.0,10.0,dtype=np.float32),
#                     shape=[3,3])
#tensor = tf.constant(np.random.rand(1024,1024))
#b = tf.cast(tensor,tf.float16)

#a = tf.cast(tensor, tf.float16)
imax = 4096

a = np.float32(np.random.rand(imax,imax))
e = tf.constant(a)
e = tf.cast(e, tf.float16)
#half-precision operation. Yeah!

print("Matrix %d x %d"%(imax,imax))
'''
ts = time.time()
dd = tf.matmul(e,e)
te = time.time()
print("Tensorflow half-precision elapsed time:",te-ts," secs")
'''

print()
ts = time.time()
for iter in range(10):
    c = tf.matmul(a,a) #add operation

te = time.time()

#print(c)
print("Tensorflow elapsed time:",te-ts," secs")

#compare to numpy

print()
ts = time.time()
for iter in range(10):
    d = np.matmul(a,a)

te = time.time()
#print(d)
print("Numpy elapsed time:",te-ts," secs")

