import tensorflow as tf
import sys
import numpy as np
import os
import time

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
    

#remove verbrose messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#eager execution allow the printing of elements
tf.compat.v1.enable_eager_execution()

imax = 4096
imax = int(input("Enter matrix row-size: "))

a = np.float32(np.random.rand(imax,imax))
acast = tf.cast(a, tf.float32) #add cast

flops = 2.0*float(imax)*float(imax)*float(imax)
itermax = 20
flops = itermax*flops
print("Matrix %d x %d"%(imax,imax))

countdown(5)

ts = time.time()
for iter in range(itermax):
    #add operation, direct numpy array is recognized
    #c = tf.matmul(a,a)

    c = tf.matmul(acast,acast) #add operation

te = time.time()

timed = te-ts
gflops = (flops/(timed))*1.0e-9
print("Tensorflow elapsed time:",timed," secs")
print("Tensorflow throughput  :",gflops," GFLOPS")

countdown(5)

#compare to numpy
print()
ts = time.time()
for iter in range(itermax):
    d = np.matmul(a,a)

te = time.time()
timed = te-ts
gflops = (flops/(timed))*1.0e-9
print("Numpy elapsed time:",timed," secs")
print("Numpy throughput  :",gflops," GFLOPS")

print(d-c)

