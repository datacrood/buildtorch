# NumPy Reshape C Code - Logic Explained

## 🎯 Main Concepts from the C Code

### 1. **Array Structure (PyArrayObject)**
In C, NumPy arrays have:
```c
- dimensions (shape)  → [2, 3, 4]
- strides            → [96, 32, 8]  (bytes to jump for each axis)
- data pointer       → actual memory location
- itemsize           → bytes per element (e.g., 8 for float64)
- flags              → C_CONTIGUOUS, F_CONTIGUOUS, etc.
```

### 2. **The Reshape Call Chain**

```
numpy.reshape(arr, shape)
    ↓
PyArray_Reshape()                    [Entry point]
    ↓
PyArray_Newshape()                   [Adds order parameter]
    ↓
_reshape_with_copy_arg()             [MAIN LOGIC]
    ↓
_attempt_nocopy_reshape()            [THE CLEVER PART]
```

---

## 🧠 Core Algorithm: `_reshape_with_copy_arg()`

This function decides: **"Can I just change the shape/strides, or must I copy data?"**

### Step-by-Step Logic:

#### **Step 1: Quick Exit**
```python
if new_shape == old_shape:
    return view  # No work needed!
```

#### **Step 2: Fix Unknown Dimension (-1)**
```python
# Example: reshape(arr, (-1, 3))
# If arr has 12 elements: -1 becomes 12/3 = 4
_fix_unknown_dimension(newshape, arr)
```

#### **Step 3: Try Zero-Copy Reshape**
```python
if _attempt_nocopy_reshape(arr, newshape):
    # SUCCESS! Just adjust strides
    return view_with_new_strides
else:
    # FAILED! Must copy and rearrange data
    return copy
```

---

## 🎓 The Smart Part: `_attempt_nocopy_reshape()`

This is the **heart of the algorithm**. It checks if we can reshape without copying.

### The Algorithm:

```
1. Remove size-1 dimensions (irrelevant for layout)
   [1, 2, 1, 3] → [2, 3]

2. Match dimension groups between old and new
   Old: [2, 3]  (product = 6)
   New: [6]     (product = 6)
   → Match!

3. Check contiguity for each group
   For C-order: stride[i] = stride[i+1] × dim[i+1]
   For F-order: stride[i+1] = stride[i] × dim[i]

4. If contiguous → Calculate new strides
   If not       → Return failure (copy needed)
```

### Example: Why (2,3) → (6,) works without copy

```
Original array (C-order):
  Shape:   [2, 3]
  Strides: [24, 8]  ← stride[0] = 3 × 8 ✓ (contiguous!)

  Memory: [0][1][2][3][4][5]  (linear in memory)

Reshape to (6,):
  Shape:   [6]
  Strides: [8]  ← Just read linearly!

  No copy needed! Same memory, different interpretation.
```

### Example: Why some reshapes need copy

```
Transposed array:
  Shape:   [2, 3]
  Strides: [8, 16]  ← stride[0] ≠ 3 × 16 ✗ (NOT contiguous!)

  Memory: [0][2][4][1][3][5]  (scrambled)

Reshape to (6,):
  Must copy and linearize: [0][1][2][3][4][5]
```

---

## 🔧 Key C Functions Explained

### `_fix_unknown_dimension()`
**Purpose:** Resolve `-1` in shape

```c
// Given: arr.size = 12, newshape = [-1, 3]
s_known = 3
i_unknown = 0
→ newshape[0] = 12 / 3 = 4
→ Result: [4, 3]
```

### `_attempt_nocopy_reshape()`
**Purpose:** Check stride compatibility

```c
Returns:
  1 (true)  → Reshape possible without copy, strides calculated
  0 (false) → Copy required
```

The checks:
```c
// C-order contiguity check:
for (ok = oi; ok < oj - 1; ok++) {
    if (oldstrides[ok] != olddims[ok+1] * oldstrides[ok+1]) {
        return 0;  // Not contiguous → need copy
    }
}

// F-order contiguity check:
if (oldstrides[ok+1] != olddims[ok] * oldstrides[ok]) {
    return 0;  // Not contiguous → need copy
}
```

---

## 💡 Why Strides Matter

**Stride** = number of bytes to jump to get to the next element along an axis

### Example: 2x3 array (8 bytes per element)

```
C-order (row-major):
  [0, 1, 2]
  [3, 4, 5]

  Strides: [24, 8]
  - Jump 24 bytes for next row (3 elements × 8 bytes)
  - Jump 8 bytes for next column

F-order (column-major):
  [0, 2, 4]
  [1, 3, 5]

  Strides: [8, 16]
  - Jump 8 bytes for next row
  - Jump 16 bytes for next column (2 elements × 8 bytes)
```

---

## 🎯 Summary: When Does Reshape Copy?

### ✅ NO COPY (View):
- Array is contiguous in the requested order
- New shape is compatible with memory layout
- Example: `(2, 3) → (6,)` on C-contiguous array

### ❌ COPY Required:
- Array is not contiguous in the requested order
- Elements need to be reordered in memory
- Example: `(2, 3) → (3, 2)` on transposed array

### 🔍 The Decision Tree:
```
reshape(arr, newshape)
  │
  ├─ Same shape? → Return view
  │
  ├─ Fix -1 dimension
  │
  ├─ Is contiguous in requested order?
  │   ├─ YES → Try stride calculation
  │   │         ├─ Success → Return view ✓
  │   │         └─ Fail → Copy
  │   └─ NO → Copy
  │
  └─ Return reshaped array
```

---

## 🚀 Performance Implications

**Why this matters:**
- Views are **instant** (just metadata change)
- Copies are **expensive** (allocate + move all data)

```python
# Fast (view):
arr = np.arange(1000000)
arr.reshape(1000, 1000)  # < 1 microsecond

# Slow (copy):
arr_t = arr.T  # Transposed (not contiguous)
arr_t.reshape(1000, 1000)  # Must copy all data
```

---

## 📚 Key Takeaways

1. **Reshape is smart**: Tries to avoid copying whenever possible
2. **Contiguity is key**: Checks if memory layout allows stride-only reshape
3. **Strides are powerful**: Different interpretations of same memory
4. **Copy is last resort**: Only when memory layout incompatible

The C code is optimized for performance, but the logic is:
*"Can I describe the new shape with just different strides, or must I physically rearrange the data?"*
