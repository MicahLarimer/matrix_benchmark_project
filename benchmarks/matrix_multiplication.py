import numpy as np
import cupy as cp
import time

#Define the size of the matrix
N = 1000 

#Define random matrices using NumPy from CuPy
A_cpu = np.random.rand(N,N).astype(np.float32)
B_cpu = np.random.rand(N,N).astype(np.float32)

#Move matrices to GPU using CuPy
A_gpu = cp.array(A_cpu)
B_gpu = cp.array(B_cpu)

#CPU Matrix Multiplication
start_time = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
end_time = time.time()
cpu_time = time.time() - start_time

#GPU Matrix Manipulation
start_time = time.time()
C_gpu = cp.dot(A_gpu, B_gpu) #Matrix multiplication on GPU using CuPy
cp.cuda.Stream.null.synchronize() # Wait for GPU computation to finish
gpu_time = time.time() - start_time

#Print results
print(f"CPU Time: {cpu_time:.4f} seconds")
print(f"GPU Time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x") #Calculate Speedup
