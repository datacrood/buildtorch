# Contiguous vs Non-Contiguous Arrays Explained

## 🎯 The Core Concept

**Contiguous** = Elements are stored **sequentially** in memory, one after another.
**Non-Contiguous** = Elements are stored with **gaps** or in a **different order** than expected.

---

## 📊 Visual Example: 2D Array

### Logical View (What You See):
```
Array: [[1, 2, 3],
        [4, 5, 6]]

Shape: (2, 3)
```

### Physical Memory View:

#### ✅ C-Contiguous (Row-Major - Default in NumPy)
```
Memory addresses: 0    8    16   24   32   40
Memory layout:   [1]  [2]  [3]  [4]  [5]  [6]
                  ↑---------↑---------↑
                  Row 0     Row 1

Reading order: Row by row (left to right, top to bottom)
Strides: [24, 8]
  - Jump 24 bytes to next row (3 elements × 8 bytes)
  - Jump 8 bytes to next column
```

#### ✅ F-Contiguous (Column-Major - Fortran order)
```
Memory addresses: 0    8    16   24   32   40
Memory layout:   [1]  [4]  [2]  [5]  [3]  [6]
                  ↑----↑    ↑----↑    ↑----↑
                  Col 0     Col 1     Col 2

Reading order: Column by column (top to bottom, left to right)
Strides: [8, 16]
  - Jump 8 bytes to next row
  - Jump 16 bytes to next column (2 elements × 8 bytes)
```

#### ❌ Non-Contiguous (e.g., Transposed)
```
Original array transposed:
Logical view: [[1, 4],
               [2, 5],
               [3, 6]]

But memory still has: [1]  [2]  [3]  [4]  [5]  [6]

To read element [1,0] (value 5):
  - Start at 0
  - Jump 8 bytes (stride[0]) → position 8, value 2 ✗

The strides don't match a contiguous layout!
Strides: [8, 24]
  - Elements are scattered, not sequential
```

---

## 🔍 Detailed Examples

### Example 1: C-Contiguous Array

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.flags['C_CONTIGUOUS'])  # True
print(arr.strides)                 # (24, 8)
```

**Memory Layout:**
```
Address:  0    8    16   24   32   40
Value:   [1]  [2]  [3]  [4]  [5]  [6]
         └─────────┘    └─────────┘
           Row 0          Row 1

To access arr[1, 2] (value 6):
  offset = 1 × 24 + 2 × 8 = 40 ✓
```

**Why it's contiguous:**
- Elements 1,2,3,4,5,6 are in order
- No gaps between elements
- Can read all data in one sequential pass

---

### Example 2: Transposed Array (Non-Contiguous)

```python
arr_t = arr.T  # Transpose

print(arr_t.flags['C_CONTIGUOUS'])  # False
print(arr_t.flags['F_CONTIGUOUS'])  # True
print(arr_t.strides)                 # (8, 24)
```

**Logical View:**
```
[[1, 4],
 [2, 5],
 [3, 6]]
```

**Memory Still Has:**
```
Address:  0    8    16   24   32   40
Value:   [1]  [2]  [3]  [4]  [5]  [6]
```

**To access arr_t[0, 1] (value 4):**
```
offset = 0 × 8 + 1 × 24 = 24 ✓

But arr_t[1, 0] (value 2):
offset = 1 × 8 + 0 × 24 = 8 ✓
```

**Why it's NOT C-contiguous:**
- For C-order, we'd read: [1, 4, 2, 5, 3, 6]
- But memory has: [1, 2, 3, 4, 5, 6]
- We must "jump around" in memory to read rows

**But it IS F-contiguous!**
- For Fortran-order, we read column-by-column
- Column 0: 1, 2, 3 ✓ (sequential at 0, 8, 16)
- Column 1: 4, 5, 6 ✓ (sequential at 24, 32, 40)

---

### Example 3: Sliced Array (Non-Contiguous)

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
arr_slice = arr[::2]  # Every other element

print(arr_slice)                      # [1, 3, 5, 7]
print(arr_slice.flags['C_CONTIGUOUS']) # False
print(arr_slice.strides)               # (16,) instead of (8,)
```

**Memory Layout:**
```
Address:  0    8    16   24   32   40   48   56
Value:   [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
         ↑         ↑         ↑         ↑
         We want these (stride = 16, skipping every other)

To access arr_slice[1] (value 3):
  offset = 1 × 16 = 16 ✓
```

**Why it's non-contiguous:**
- Selected elements: 1, 3, 5, 7
- Memory has gaps: [1] __ [3] __ [5] __ [7]
- Stride is 16 (skipping 8 bytes each time)

---

## 🧮 The Contiguity Check

### C-Contiguous Check:
```python
def is_c_contiguous(shape, strides, itemsize):
    """
    For C-order, innermost dimension should have stride = itemsize
    Each outer dimension: stride[i] = stride[i+1] × shape[i+1]
    """
    if strides[-1] != itemsize:
        return False

    for i in range(len(shape) - 2, -1, -1):
        if strides[i] != strides[i + 1] * shape[i + 1]:
            return False

    return True

# Example: shape=(2,3), strides=(24,8), itemsize=8
# strides[-1] = 8 = itemsize ✓
# strides[0] = 24 = strides[1] × shape[1] = 8 × 3 ✓
# → C-contiguous!
```

### F-Contiguous Check:
```python
def is_f_contiguous(shape, strides, itemsize):
    """
    For F-order, outermost dimension should have stride = itemsize
    Each inner dimension: stride[i+1] = stride[i] × shape[i]
    """
    if strides[0] != itemsize:
        return False

    for i in range(len(shape) - 1):
        if strides[i + 1] != strides[i] * shape[i]:
            return False

    return True

# Example: shape=(2,3), strides=(8,16), itemsize=8
# strides[0] = 8 = itemsize ✓
# strides[1] = 16 = strides[0] × shape[0] = 8 × 2 ✓
# → F-contiguous!
```

---

## ⚡ Why Contiguity Matters

### 1. **Performance**

#### Contiguous (Fast):
```python
arr = np.arange(1000000)  # Contiguous
sum = arr.sum()  # CPU can read sequentially
# → Cache-friendly, SIMD vectorization possible
```

#### Non-Contiguous (Slower):
```python
arr_t = arr.reshape(1000, 1000).T  # Non-contiguous
sum = arr_t.sum()  # CPU must jump around
# → Cache misses, no vectorization
```

**Speed difference:** Can be **10-100x slower**!

### 2. **Memory Operations**

```python
# Contiguous → Can use fast memcpy
arr_copy = arr.copy()  # Fast!

# Non-contiguous → Must copy element by element
arr_t_copy = arr_t.copy()  # Slower
```

### 3. **C/Fortran Interop**

```python
# Many C libraries expect contiguous arrays
import ctypes

# This works:
arr_c = np.ascontiguousarray(arr)
c_func(arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# This might fail or give wrong results:
# c_func(arr_t.ctypes.data_as(...))  # Non-contiguous!
```

---

## 🎓 How Arrays Become Non-Contiguous

### 1. **Transpose**
```python
arr = np.array([[1, 2], [3, 4]])  # C-contiguous
arr_t = arr.T                      # NOT C-contiguous (but F-contiguous)
```

### 2. **Slicing with Steps**
```python
arr = np.arange(10)      # Contiguous
arr_slice = arr[::2]     # Non-contiguous (gaps)
```

### 3. **Fancy Indexing** (sometimes)
```python
arr = np.arange(10)
arr_fancy = arr[[0, 5, 3]]  # May be non-contiguous
```

### 4. **Broadcasted Operations** (views)
```python
arr = np.array([1, 2, 3])
arr_broadcast = np.broadcast_to(arr, (3, 3))  # Non-contiguous
```

---

## 🔧 Making Arrays Contiguous

### Method 1: `np.ascontiguousarray()`
```python
arr_t = arr.T  # Non-contiguous
arr_c = np.ascontiguousarray(arr_t)  # Contiguous copy
```

### Method 2: `.copy()`
```python
arr_copy = arr_t.copy()  # Always contiguous (default C-order)
arr_copy_f = arr_t.copy(order='F')  # F-contiguous
```

### Method 3: `np.asfortranarray()`
```python
arr_f = np.asfortranarray(arr)  # F-contiguous
```

---

## 📊 Quick Reference Table

| Operation | Result | Why |
|-----------|--------|-----|
| `np.arange(10)` | C-contiguous | Created sequentially |
| `arr.reshape(2, 5)` | Usually contiguous | Can rearrange strides |
| `arr.T` | NOT C-contiguous | Elements reordered logically |
| `arr[::2]` | Non-contiguous | Skips elements (gaps) |
| `arr[:, 0]` | Non-contiguous | Column slice (jumps) |
| `arr[0, :]` | Contiguous | Row slice (sequential) |
| `arr.copy()` | C-contiguous | Fresh memory layout |

---

## 🎯 Key Takeaways

1. **Contiguous = Sequential in memory** (fast to access)
2. **Non-contiguous = Requires jumps** (slower access)
3. **Two types of contiguity:**
   - **C-order**: Row-major (NumPy default)
   - **F-order**: Column-major (MATLAB/Fortran)
4. **Strides tell you the memory layout:**
   - Contiguous: Strides follow predictable pattern
   - Non-contiguous: Strides are "weird"
5. **Performance matters:**
   - Contiguous: 10-100x faster for many operations
   - Non-contiguous: Cache-unfriendly

**Rule of thumb:** If performance matters, keep arrays contiguous!

---

## 🧪 Test Yourself

```python
import numpy as np

# Which are contiguous?
a = np.array([[1, 2], [3, 4]])          # C-contiguous ✓
b = a.T                                  # F-contiguous ✓ (NOT C-contiguous)
c = a[::2]                               # Non-contiguous ✗
d = a[:, 0]                              # Non-contiguous ✗ (column)
e = a[0, :]                              # C-contiguous ✓ (row)
f = np.ascontiguousarray(b)              # C-contiguous ✓ (forced)

# Check:
print(a.flags['C_CONTIGUOUS'])  # True
print(b.flags['C_CONTIGUOUS'])  # False
print(b.flags['F_CONTIGUOUS'])  # True
print(c.flags['C_CONTIGUOUS'])  # False
print(d.flags['C_CONTIGUOUS'])  # False
print(e.flags['C_CONTIGUOUS'])  # True
print(f.flags['C_CONTIGUOUS'])  # True
```
