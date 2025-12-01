"""
Interactive demonstration of contiguous vs non-contiguous arrays
"""

import numpy as np


def print_array_info(arr, name="Array"):
    """Print detailed information about an array's memory layout"""
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    print(f"Shape:          {arr.shape}")
    print(f"Strides:        {arr.strides}")
    print(f"Item size:      {arr.itemsize} bytes")
    print(f"C-contiguous:   {arr.flags['C_CONTIGUOUS']} {'✓' if arr.flags['C_CONTIGUOUS'] else '✗'}")
    print(f"F-contiguous:   {arr.flags['F_CONTIGUOUS']} {'✓' if arr.flags['F_CONTIGUOUS'] else '✗'}")
    print(f"Data pointer:   {arr.ctypes.data}")
    print(f"\nLogical view:\n{arr}")


def visualize_memory_layout(arr, name="Array"):
    """Visualize how array elements are stored in memory"""
    print(f"\n{'='*70}")
    print(f"🔬 Memory Layout: {name}")
    print(f"{'='*70}")

    # Flatten to see actual memory order
    if arr.flags['C_CONTIGUOUS']:
        flat = arr.ravel('C')
        order = "C-order (row-major)"
    elif arr.flags['F_CONTIGUOUS']:
        flat = arr.ravel('F')
        order = "F-order (column-major)"
    else:
        flat = arr.ravel('K')  # Keep order
        order = "Non-contiguous"

    print(f"Memory order: {order}")
    print(f"\nMemory addresses (relative to base):")

    # Show first few elements with their memory positions
    for i in range(min(len(flat), 10)):
        offset = i * arr.itemsize
        print(f"  Offset {offset:3d}: value = {flat[i]}")

    if len(flat) > 10:
        print(f"  ... ({len(flat) - 10} more elements)")


def demonstrate_contiguous():
    """Show examples of contiguous arrays"""
    print("\n" + "="*70)
    print("✅ CONTIGUOUS ARRAYS")
    print("="*70)

    # Example 1: Basic array (C-contiguous)
    print("\n1️⃣ Basic array creation (C-contiguous by default)")
    arr1 = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=np.int64)
    print_array_info(arr1, "Basic 2D Array")
    visualize_memory_layout(arr1, "Basic 2D Array")

    # Example 2: F-contiguous array
    print("\n\n2️⃣ Fortran-order array (F-contiguous)")
    arr2 = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=np.int64, order='F')
    print_array_info(arr2, "Fortran-order Array")
    visualize_memory_layout(arr2, "Fortran-order Array")

    # Example 3: Reshaped array
    print("\n\n3️⃣ Reshaped array (stays contiguous)")
    arr3 = np.arange(6, dtype=np.int64).reshape(2, 3)
    print_array_info(arr3, "Reshaped Array")


def demonstrate_non_contiguous():
    """Show examples of non-contiguous arrays"""
    print("\n\n" + "="*70)
    print("❌ NON-CONTIGUOUS ARRAYS")
    print("="*70)

    # Example 1: Transposed array
    print("\n1️⃣ Transposed array")
    arr1 = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=np.int64)
    arr1_t = arr1.T
    print_array_info(arr1, "Original Array")
    print_array_info(arr1_t, "Transposed Array")

    print("\n💡 Notice:")
    print(f"   - Original strides: {arr1.strides} (C-contiguous)")
    print(f"   - Transposed strides: {arr1_t.strides} (NOT C-contiguous)")
    print(f"   - Same data pointer: {arr1.ctypes.data} == {arr1_t.ctypes.data}")
    print(f"   - Just different interpretation!")

    # Example 2: Sliced array
    print("\n\n2️⃣ Sliced array (every other element)")
    arr2 = np.arange(10, dtype=np.int64)
    arr2_slice = arr2[::2]
    print_array_info(arr2, "Original Array")
    print_array_info(arr2_slice, "Sliced Array (every 2nd element)")

    print("\n💡 Notice:")
    print(f"   - Original stride: {arr2.strides[0]} (contiguous)")
    print(f"   - Sliced stride: {arr2_slice.strides[0]} (skipping elements)")
    print(f"   - Stride doubled because we skip every other element!")

    # Example 3: Column slice
    print("\n\n3️⃣ Column slice from 2D array")
    arr3 = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=np.int64)
    col = arr3[:, 1]  # Second column
    print_array_info(arr3, "Original 2D Array")
    print_array_info(col, "Column Slice [:, 1]")

    print("\n💡 Notice:")
    print(f"   - Column values: {col} (elements 2, 5)")
    print(f"   - Stride: {col.strides[0]} bytes")
    print(f"   - Must jump {col.strides[0]} bytes to get from 2 to 5")
    print(f"   - Not sequential in memory!")


def demonstrate_making_contiguous():
    """Show how to make arrays contiguous"""
    print("\n\n" + "="*70)
    print("🔧 MAKING ARRAYS CONTIGUOUS")
    print("="*70)

    # Create non-contiguous array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.int64)
    arr_t = arr.T

    print("\n1️⃣ Using np.ascontiguousarray()")
    arr_c = np.ascontiguousarray(arr_t)
    print_array_info(arr_t, "Original (Non-contiguous)")
    print_array_info(arr_c, "After ascontiguousarray()")

    print("\n💡 Notice:")
    print(f"   - Data pointer changed: {arr_t.ctypes.data} → {arr_c.ctypes.data}")
    print(f"   - New memory allocated!")
    print(f"   - Strides now follow C-order pattern")

    print("\n2️⃣ Using .copy()")
    arr_copy = arr_t.copy()
    print_array_info(arr_copy, "After .copy()")

    print("\n3️⃣ Using .copy(order='F') for F-contiguous")
    arr_f = arr_t.copy(order='F')
    print_array_info(arr_f, "After .copy(order='F')")


def demonstrate_performance():
    """Show performance difference"""
    print("\n\n" + "="*70)
    print("⚡ PERFORMANCE COMPARISON")
    print("="*70)

    import time

    size = 10000
    iterations = 1000

    # Contiguous array
    arr_c = np.arange(size * size, dtype=np.float64).reshape(size, size)

    # Non-contiguous array (transposed)
    arr_nc = arr_c.T

    # Time contiguous sum
    start = time.time()
    for _ in range(iterations):
        _ = arr_c.sum()
    time_c = time.time() - start

    # Time non-contiguous sum
    start = time.time()
    for _ in range(iterations):
        _ = arr_nc.sum()
    time_nc = time.time() - start

    print(f"\nArray size: {size} × {size} = {size*size:,} elements")
    print(f"Iterations: {iterations}")
    print(f"\nContiguous array sum:     {time_c:.4f} seconds")
    print(f"Non-contiguous array sum: {time_nc:.4f} seconds")
    print(f"Speedup: {time_nc/time_c:.2f}x faster when contiguous!")

    print("\n💡 Why the difference?")
    print("   - Contiguous: CPU can read sequentially (cache-friendly)")
    print("   - Non-contiguous: CPU must jump around (cache misses)")


def check_contiguity_math():
    """Demonstrate the mathematical check for contiguity"""
    print("\n\n" + "="*70)
    print("🧮 CONTIGUITY CHECK (THE MATH)")
    print("="*70)

    arr = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.int64)

    print(f"\nArray shape: {arr.shape}")
    print(f"Strides: {arr.strides}")
    print(f"Item size: {arr.itemsize}")

    print("\n📐 C-Contiguous Check:")
    print(f"   1. Last stride == itemsize?")
    print(f"      {arr.strides[-1]} == {arr.itemsize} → {arr.strides[-1] == arr.itemsize} ✓")

    print(f"\n   2. Each stride[i] == stride[i+1] × shape[i+1]?")
    for i in range(len(arr.shape) - 1):
        expected = arr.strides[i + 1] * arr.shape[i + 1]
        actual = arr.strides[i]
        match = "✓" if actual == expected else "✗"
        print(f"      stride[{i}] = {actual}, expected = {arr.strides[i+1]} × {arr.shape[i+1]} = {expected} → {match}")

    print(f"\n   → C-Contiguous: {arr.flags['C_CONTIGUOUS']}")

    # Now check transposed
    arr_t = arr.T
    print(f"\n\nTransposed array shape: {arr_t.shape}")
    print(f"Strides: {arr_t.strides}")

    print("\n📐 F-Contiguous Check:")
    print(f"   1. First stride == itemsize?")
    print(f"      {arr_t.strides[0]} == {arr_t.itemsize} → {arr_t.strides[0] == arr_t.itemsize} ✓")

    print(f"\n   2. Each stride[i+1] == stride[i] × shape[i]?")
    for i in range(len(arr_t.shape) - 1):
        expected = arr_t.strides[i] * arr_t.shape[i]
        actual = arr_t.strides[i + 1]
        match = "✓" if actual == expected else "✗"
        print(f"      stride[{i+1}] = {actual}, expected = {arr_t.strides[i]} × {arr_t.shape[i]} = {expected} → {match}")

    print(f"\n   → F-Contiguous: {arr_t.flags['F_CONTIGUOUS']}")


if __name__ == "__main__":
    print("\n" + "🎓 " * 35)
    print("NUMPY CONTIGUOUS vs NON-CONTIGUOUS ARRAYS - INTERACTIVE DEMO")
    print("🎓 " * 35)

    demonstrate_contiguous()
    demonstrate_non_contiguous()
    demonstrate_making_contiguous()
    check_contiguity_math()
    demonstrate_performance()

    print("\n\n" + "="*70)
    print("✅ SUMMARY")
    print("="*70)
    print("""
Contiguous Arrays:
  ✓ Elements stored sequentially in memory
  ✓ Fast to access (cache-friendly)
  ✓ Can use SIMD vectorization
  ✓ Required by many C/Fortran libraries

Non-Contiguous Arrays:
  ✗ Elements scattered or require jumps
  ✗ Slower to access (cache misses)
  ✗ May need copying before passing to C code
  ✗ Common after: transpose, slicing, fancy indexing

Key Operations:
  • np.ascontiguousarray(arr) → Make C-contiguous
  • arr.copy()                → Make C-contiguous copy
  • arr.copy(order='F')       → Make F-contiguous copy
  • arr.flags['C_CONTIGUOUS'] → Check if C-contiguous
  • arr.flags['F_CONTIGUOUS'] → Check if F-contiguous

Remember: When performance matters, keep it contiguous! 🚀
""")
    print("="*70)
