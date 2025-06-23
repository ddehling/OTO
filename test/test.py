import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

def benchmark(sizes, func_np, func_cp, operation_name):
    """Run benchmark for various array sizes."""
    np_times = []
    cp_times = []
    
    for size in sizes:
        # NumPy benchmark
        start = time.time()
        func_np(size)
        np_time = time.time() - start
        np_times.append(np_time)
        
        # CuPy benchmark
        start = time.time()
        func_cp(size)
        cp.cuda.Stream.null.synchronize()  # Make sure GPU operations are completed
        cp_time = time.time() - start
        cp_times.append(cp_time)
        
        # Calculate speedup with safeguard against division by zero
        if cp_time > 0:
            speedup = np_time / cp_time
            print(f"{operation_name} - Size: {size}, NumPy: {np_time:.6f}s, CuPy: {cp_time:.6f}s, Speedup: {speedup:.2f}x")
        else:
            print(f"{operation_name} - Size: {size}, NumPy: {np_time:.6f}s, CuPy: {cp_time:.6f}s, Speedup: inf (CuPy time too small to measure)")
    
    return np_times, cp_times

# Define operations to benchmark
def matrix_mul_np(size):
    a = np.random.random((size, size))
    b = np.random.random((size, size))
    return np.dot(a, b)

def matrix_mul_cp(size):
    a = cp.random.random((size, size))
    b = cp.random.random((size, size))
    return cp.dot(a, b)

def elementwise_op_np(size):
    a = np.random.random((size, size))
    b = np.random.random((size, size))
    return a * b + np.sin(a)

def elementwise_op_cp(size):
    a = cp.random.random((size, size))
    b = cp.random.random((size, size))
    return a * b + cp.sin(a)

def reduction_np(size):
    a = np.random.random((size, size))
    return np.sum(a, axis=1)

def reduction_cp(size):
    a = cp.random.random((size, size))
    return cp.sum(a, axis=1)

# Run benchmarks
sizes = [500, 1000, 2000, 4000, 6000]
operations = [
    (matrix_mul_np, matrix_mul_cp, "Matrix Multiplication"),
    (elementwise_op_np, elementwise_op_cp, "Elementwise Operations"),
    (reduction_np, reduction_cp, "Reduction (Sum)")
]

results = {}

print("Starting benchmarks...\n")

# First warm up the GPU
warm_up = cp.random.random((1000, 1000))
cp.dot(warm_up, warm_up)
cp.cuda.Stream.null.synchronize()

for np_func, cp_func, name in operations:
    print(f"\n--- {name} Benchmark ---")
    np_times, cp_times = benchmark(sizes, np_func, cp_func, name)
    results[name] = (np_times, cp_times)

# Plot results
plt.figure(figsize=(15, 10))

for i, (name, (np_times, cp_times)) in enumerate(results.items(), 1):
    plt.subplot(len(operations), 1, i)
    plt.plot(sizes, np_times, 'o-', label='NumPy')
    plt.plot(sizes, cp_times, 'o-', label='CuPy')
    plt.title(f'{name} - NumPy vs CuPy')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('numpy_vs_cupy_benchmark.png')
plt.show()

# Calculate average speedup
for name, (np_times, cp_times) in results.items():
    speedups = [n/c if c > 0 else float('inf') for n, c in zip(np_times, cp_times)]
    # Filter out infinities for average calculation
    finite_speedups = [s for s in speedups if s != float('inf')]
    if finite_speedups:
        avg_speedup = sum(finite_speedups) / len(finite_speedups)
        print(f"\n{name} - Average Speedup: {avg_speedup:.2f}x")
    else:
        print(f"\n{name} - Average Speedup: Could not calculate (all CuPy times were zero)")
