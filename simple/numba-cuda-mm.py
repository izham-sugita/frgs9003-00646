from __future__ import division
from numba import cuda, float32
import numpy
import math
import time

#set the environment variables
# $export CUDA_HOME=/your/local/cuda/directory/with/version
# before running this script

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

# The data array
#A = numpy.full((TPB*2, TPB*3), 3, numpy.float) # [32 x 48] matrix containing all 3's
#B = numpy.full((TPB*3, TPB*1), 4, numpy.float) # [48 x 16] matrix containing all 4's

# The data array
imax = int(input("Enter imax:"))
#imax = 1024
jmax = imax
print("A and B are %d x %d matrix"%(imax,jmax))
A = numpy.full((imax, jmax), 3, numpy.float32) # [32 x 48] matrix containing all 3's
B = numpy.full((jmax, imax), 4, numpy.float32) # [48 x 16] matrix containing all 4's

A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
#C_global_mem = cuda.device_array((TPB*2, TPB*1)) # [32 x 16] matrix result
C_global_mem = cuda.device_array((imax,imax )) # [32 x 16] matrix result

# Configure the blocks
threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

ts = time.time()
# Start the kernel 
fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
res = C_global_mem.copy_to_host()
te = time.time()
elapsed_time = te-ts
flops = 2.0*imax*jmax*jmax
gflops = (flops/elapsed_time)*1.0e-9
print("Numba-CUDA Throughput: %.4f GFLOPS"%gflops)

print()
#matmul with numpy
ts = time.time()
C_host = numpy.matmul(A,B)
te = time.time()
elapsed_time = te-ts
gflops = (flops/elapsed_time)*1.0e-9
print("Numpy Throughput: %.4f GFLOPS"%gflops)

print(res-C_host)
