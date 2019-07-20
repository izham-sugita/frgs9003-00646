import time
import tensorflow as tf
import numpy as np

def np_mm(x):
    iter = 10
    start=time.time()
    for loop in range(iter):
        a = np.matmul(x,x)

    elapsed=time.time() -start

    print("%d loop: %.4f secs"%(iter,elapsed))
    

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

#a = np.ndarray(shape=(1000,1000), dtype=np.float32)
a = np.float32(np.random.rand(2048,2048))

x = tf.constant(a)

time_matmul(x)

np_mm(a)

# Force execution on CPU
#print("On CPU:")
#with tf.device("CPU:0"):
#  x = tf.random.uniform([1000, 1000])
#  assert x.device.endswith("CPU:0")
#  time_matmul(x)

# Force execution on GPU #0 if available
#if tf.test.is_gpu_available():
#  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
#    x = tf.random.uniform([1000, 1000])
#    assert x.device.endswith("GPU:0")
#    time_matmul(x)
