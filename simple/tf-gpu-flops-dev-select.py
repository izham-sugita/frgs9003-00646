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


imax = int(input("Enter matrix row-size: "))
a = np.float32(np.random.rand(imax,imax))

# dev = string variable; use for input

dev =input("Select device: GPU or CPU ")
if dev == 'CPU' or dev =='cpu':
    print("Selected cpu")
    dev = dev.lower()
    dev = "/"+dev+":0"
    print(dev)

if dev =='GPU' or dev =='gpu':
    print("Selected gpu")
    num = input("Device number(0-2): ")
    dev = dev.upper()
    dev= "/device:"+dev+":"+num
    print(dev)

'''
if num == 0 or num == 1:
    #create on device/cpu
    with tf.device(dev):
        acast = tf.cast(a, tf.float32) #add cast
else:
    acast = a
'''
with tf.device(dev):
    acast = tf.cast(a, tf.float32) #cast to cpu
    
flops = 2.0*float(imax)*float(imax)*float(imax)
itermax = 20
flops = itermax*flops
print("Matrix %d x %d"%(imax,imax))

countdown(5)

ts = time.time()

with tf.device(dev):
    for iter in range(itermax):
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

#print(d-c)

