from typing import List, Optional
import numpy as np

class Tensor:
    def __init__(self, data=None, requires_grad=False):
        # self.data = self._data_to_numpy(data)
        # Float32 to keep consistency. 
        self.data = np.array(data, dtype=np.float32) #Since numpy already handles type conversion, let's use it to keep code simple.
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
  
    def _data_to_numpy(self, data):
        import numpy as np
        if isinstance(data, list):
            return np.array(data)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, size={self.size}, dtype={self.dtype}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        """Custom addition for Tensor objects to create a new Tensor object"""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
            # return self.data + other.data
        elif isinstance(other, (int, float)): #Broadcasting support
            return Tensor(self.data + other)
        else:
            raise ValueError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
    
    def __sub__(self, other):
        """Custom subtraction for Tensor objects to create a new Tensor object"""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        elif isinstance(other, (int, float)): #Broadcasting support
            return Tensor(self.data - other)
        else:
            raise ValueError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")
        
    def __mul__(self, other):
        """Custom multiplication for Tensor objects to create a new Tensor object"""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        elif isinstance(other, (int, float)): #Broadcasting support
            return Tensor(self.data * other)
        else:
            raise ValueError(f"Unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        elif isinstance(other, (int, float)): #Broadcasting support
            if other == 0: #Can be floating point zero?
                raise ZeroDivisionError("Cannot divide by zero")
            return Tensor(self.data / other)
        else:
            raise ValueError(f"Unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

    def matmul(self, other):
        """For dealing with matrix, numpy uses row first operations i.e., 
        if a = [[1,2,3], [4,5,6]] -> a.flat will return flat array in order [1,2,3,4,5,6] instead of [1,4,2,5,3,6]"""
        if isinstance(other, Tensor):
            # Shape will be dynamic, can be one, two, three and so on...
            # Note: matrix multiplication in mathematics is only allowed by d dimemnsions. But tensor = multiple dimensions. Check defination of matrix.
            # So, broadcast rules is applied to tensors, to define what can be multiplies similar to a matrix. 
            # The order of matrix multiplication matters.
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix dimensions do not match")
            new_matrix = []

            # for val in self.data.flat:
            #     for first_mat_row in range(self.shape[0]):
            #         for second_mat_column in range(other.shape[1]):

    
if __name__ == "__main__":
    h = Tensor([1,2,3])
    print(h)
    # z = Tensor([4,5,6])
    # matrix = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    # vector = Tensor([10, 20])          # Shape: (2,)
    # result = matrix + vector           # Broadcasting: (2,2) + (2,) → (2,2)
    # print(result.data)
