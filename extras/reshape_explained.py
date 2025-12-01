"""
Python translation of NumPy's reshape C implementation
This is a simplified version to understand the algorithm
"""

class MockArray:
    """Simplified ndarray structure"""
    def __init__(self, shape, strides, data, itemsize=8, is_c_contiguous=True, is_f_contiguous=False):
        self.shape = list(shape)
        self.strides = list(strides)
        self.data = data  # Just a reference
        self.itemsize = itemsize
        self.is_c_contiguous = is_c_contiguous
        self.is_f_contiguous = is_f_contiguous
        self.ndim = len(shape)

    def size(self):
        """Total number of elements"""
        size = 1
        for dim in self.shape:
            size *= dim
        return size


# Constants
NPY_CORDER = 0
NPY_FORTRANORDER = 1
NPY_ANYORDER = 2
NPY_KEEPORDER = 3

NPY_COPY_ALWAYS = 0
NPY_COPY_IF_NEEDED = 1
NPY_COPY_NEVER = 2

NPY_MAXDIMS = 32


def PyArray_Reshape(array, new_shape):
    """
    Main entry point for reshape (corresponds to numpy.reshape())

    Args:
        array: MockArray object
        new_shape: tuple of new dimensions

    Returns:
        Reshaped array (view or copy)
    """
    # Convert shape tuple to list
    newdims = list(new_shape)

    # Call the main reshape function with C-order
    return PyArray_Newshape(array, newdims, NPY_CORDER)


def PyArray_Newshape(array, newdims, order):
    """
    Creates new shape for array
    Uses copy-if-needed strategy
    """
    return _reshape_with_copy_arg(array, newdims, order, NPY_COPY_IF_NEEDED)


def _reshape_with_copy_arg(array, newdims, order, copy_mode):
    """
    Core reshape implementation

    This is WHERE THE MAGIC HAPPENS!

    Args:
        array: MockArray to reshape
        newdims: list of new dimensions
        order: memory order (C or Fortran)
        copy_mode: whether to copy (ALWAYS/IF_NEEDED/NEVER)

    Returns:
        Reshaped array
    """
    ndim = len(newdims)
    dimensions = list(newdims)

    # Step 1: Handle ANYORDER
    if order == NPY_ANYORDER:
        order = NPY_FORTRANORDER if array.is_f_contiguous else NPY_CORDER
    elif order == NPY_KEEPORDER:
        raise ValueError("order 'K' is not permitted for reshaping")

    # Step 2: Quick check - is reshape even needed?
    if ndim == array.ndim and copy_mode != NPY_COPY_ALWAYS:
        same = True
        for i in range(ndim):
            if array.shape[i] != dimensions[i]:
                same = False
                break

        if same:
            print("✓ Shape already matches - returning view")
            return array  # Just return a view

    # Step 3: Fix any -1 dimensions (unknown dimension)
    if not _fix_unknown_dimension(dimensions, array):
        raise ValueError("Cannot fix unknown dimension")

    print(f"After fixing -1: new shape = {dimensions}")

    # Step 4: Handle copy modes
    strides = None
    newstrides = [0] * NPY_MAXDIMS

    if copy_mode == NPY_COPY_ALWAYS:
        print("✗ COPY_ALWAYS mode - creating copy")
        array = _copy_array(array, order)
    else:
        # Step 5: Try to avoid copying!
        # Check if we need to reorder memory
        needs_copy = False

        if order == NPY_CORDER and not array.is_c_contiguous:
            needs_copy = True
        elif order == NPY_FORTRANORDER and not array.is_f_contiguous:
            needs_copy = True

        if needs_copy:
            print("→ Array is not contiguous in requested order")
            print("  Attempting no-copy reshape...")

            # TRY THE SMART ALGORITHM!
            success = _attempt_nocopy_reshape(
                array, ndim, dimensions, newstrides,
                order == NPY_FORTRANORDER
            )

            if success:
                print("✓ No-copy reshape SUCCESS! Using new strides")
                strides = newstrides[:ndim]
            else:
                print("✗ No-copy reshape failed")
                if copy_mode == NPY_COPY_NEVER:
                    raise ValueError("Unable to avoid creating a copy while reshaping")
                else:
                    print("  Creating copy...")
                    array = _copy_array(array, order)

    # Step 6: Create the reshaped array
    # Update flags based on order
    if ndim > 1:
        if order == NPY_FORTRANORDER:
            array.is_c_contiguous = False
            array.is_f_contiguous = True
        else:
            array.is_c_contiguous = True
            array.is_f_contiguous = False

    # Create new array descriptor with new shape and strides
    ret = MockArray(
        shape=dimensions,
        strides=strides if strides else array.strides[:ndim],
        data=array.data,
        itemsize=array.itemsize,
        is_c_contiguous=array.is_c_contiguous,
        is_f_contiguous=array.is_f_contiguous
    )

    return ret


def _attempt_nocopy_reshape(array, newnd, newdims, newstrides, is_f_order):
    """
    THE CLEVER ALGORITHM!

    Attempts to reshape without copying by recalculating strides.

    Returns:
        True if successful (fills newstrides)
        False if copy is needed

    Algorithm:
        1. Remove size-1 dimensions (they don't affect layout)
        2. Match groups of old and new dimensions
        3. Check if each group is contiguous
        4. Calculate new strides if contiguous
    """
    print("\n  === Attempting No-Copy Reshape ===")

    # Step 1: Remove axes with dimension 1 from old array
    olddims = []
    oldstrides = []
    for i in range(array.ndim):
        if array.shape[i] != 1:
            olddims.append(array.shape[i])
            oldstrides.append(array.strides[i])

    oldnd = len(olddims)
    print(f"  After removing size-1: old shape={olddims}, strides={oldstrides}")

    # Step 2: Match dimension groups
    oi = 0  # old index start
    oj = 1  # old index end
    ni = 0  # new index start
    nj = 1  # new index end

    while ni < newnd and oi < oldnd:
        np_prod = newdims[ni]  # new product
        op_prod = olddims[oi]  # old product

        # Expand groups until products match
        while np_prod != op_prod:
            if np_prod < op_prod:
                # Need more new dimensions
                if nj >= newnd:
                    break
                np_prod *= newdims[nj]
                nj += 1
            else:
                # Need more old dimensions
                if oj >= oldnd:
                    break
                op_prod *= olddims[oj]
                oj += 1

        if np_prod != op_prod:
            print(f"  ✗ Product mismatch: {np_prod} != {op_prod}")
            return False

        print(f"  Matched group: old[{oi}:{oj}] = new[{ni}:{nj}], product={np_prod}")

        # Step 3: Check if old axes can be combined (contiguous check)
        for ok in range(oi, oj - 1):
            if is_f_order:
                # Fortran order: stride[k+1] should = dim[k] * stride[k]
                expected = olddims[ok] * oldstrides[ok]
                if oldstrides[ok + 1] != expected:
                    print(f"  ✗ Not F-contiguous: stride[{ok+1}]={oldstrides[ok+1]} != {expected}")
                    return False
            else:
                # C order: stride[k] should = dim[k+1] * stride[k+1]
                expected = olddims[ok + 1] * oldstrides[ok + 1]
                if oldstrides[ok] != expected:
                    print(f"  ✗ Not C-contiguous: stride[{ok}]={oldstrides[ok]} != {expected}")
                    return False

        print(f"  ✓ Group is contiguous!")

        # Step 4: Calculate new strides for this group
        if is_f_order:
            # Fortran order: propagate forward
            newstrides[ni] = oldstrides[oi]
            for nk in range(ni + 1, nj):
                newstrides[nk] = newstrides[nk - 1] * newdims[nk - 1]
        else:
            # C order: propagate backward
            newstrides[nj - 1] = oldstrides[oj - 1]
            for nk in range(nj - 2, ni - 1, -1):
                newstrides[nk] = newstrides[nk + 1] * newdims[nk + 1]

        print(f"  New strides[{ni}:{nj}] = {newstrides[ni:nj]}")

        # Move to next group
        ni = nj
        nj = ni + 1
        oi = oj
        oj = oi + 1

    # Step 5: Handle trailing size-1 dimensions in new shape
    if ni >= 1:
        last_stride = newstrides[ni - 1]
        if is_f_order:
            last_stride *= newdims[ni - 1]
    else:
        last_stride = array.itemsize

    for nk in range(ni, newnd):
        newstrides[nk] = last_stride

    print(f"  ✓ SUCCESS! New strides = {newstrides[:newnd]}")
    return True


def _fix_unknown_dimension(newshape, array):
    """
    Handles -1 in reshape (unknown dimension)

    Example: reshape(12,) to (-1, 3) → becomes (4, 3)

    Returns:
        True if successful (modifies newshape in place)
        False if error
    """
    n = len(newshape)
    s_original = array.size()
    s_known = 1
    i_unknown = -1

    # Find the -1 and calculate known product
    for i in range(n):
        if newshape[i] < 0:
            if i_unknown == -1:
                i_unknown = i
            else:
                print("ERROR: Can only specify one unknown dimension")
                return False
        else:
            s_known *= newshape[i]

    # Calculate the unknown dimension
    if i_unknown >= 0:
        if s_known == 0 or s_original % s_known != 0:
            print(f"ERROR: Cannot reshape size {s_original} into shape {newshape}")
            return False
        newshape[i_unknown] = s_original // s_known
    else:
        # No unknown dimension, just verify sizes match
        if s_original != s_known:
            print(f"ERROR: Size mismatch {s_original} != {s_known}")
            return False

    return True


def _copy_array(array, order):
    """Simulates creating a copy in specified order"""
    print(f"  Creating {'Fortran' if order == NPY_FORTRANORDER else 'C'}-order copy")
    # In real implementation, this copies and reorders data
    new_array = MockArray(
        shape=array.shape[:],
        strides=array.strides[:],
        data=f"{array.data}_copy",
        itemsize=array.itemsize,
        is_c_contiguous=(order == NPY_CORDER),
        is_f_contiguous=(order == NPY_FORTRANORDER)
    )
    return new_array


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NumPy Reshape Algorithm - Python Translation")
    print("=" * 70)

    # Example 1: Simple reshape that can avoid copy
    print("\n\n📝 Example 1: Reshape (2,3) → (6,)")
    print("-" * 70)
    arr1 = MockArray(
        shape=(2, 3),
        strides=(24, 8),  # C-contiguous: stride[0] = 3 * 8 = 24
        data="data1",
        is_c_contiguous=True
    )
    print(f"Original: shape={arr1.shape}, strides={arr1.strides}")
    result1 = PyArray_Reshape(arr1, (6,))
    print(f"Result: shape={result1.shape}, strides={result1.strides}")

    # Example 2: Reshape with unknown dimension
    print("\n\n📝 Example 2: Reshape (2,3) → (-1, 2) [unknown dimension]")
    print("-" * 70)
    arr2 = MockArray(
        shape=(2, 3),
        strides=(24, 8),
        data="data2",
        is_c_contiguous=True
    )
    print(f"Original: shape={arr2.shape}, strides={arr2.strides}")
    result2 = PyArray_Reshape(arr2, (-1, 2))
    print(f"Result: shape={result2.shape}, strides={result2.strides}")

    # Example 3: Reshape that requires copy
    print("\n\n📝 Example 3: Non-contiguous reshape (2,3) → (3,2)")
    print("-" * 70)
    arr3 = MockArray(
        shape=(2, 3),
        strides=(8, 16),  # NOT C-contiguous (transposed-like)
        data="data3",
        is_c_contiguous=False
    )
    print(f"Original: shape={arr3.shape}, strides={arr3.strides}")
    result3 = PyArray_Reshape(arr3, (3, 2))
    print(f"Result: shape={result3.shape}, strides={result3.strides}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Reshape tries to avoid copying by recalculating strides")
    print("2. It checks if memory layout is compatible with new shape")
    print("3. Only copies if absolutely necessary (non-contiguous data)")
    print("4. Handles -1 by calculating from total size")
    print("=" * 70)
