def flatten_recursive(seq):
    for item in seq:
        if isinstance(item, (list, tuple)):
            yield from flatten_recursive(item)
        else:
            yield item
            
nested = [1, [2, [3, 4], 5], (6, 7, [8, 9, [10]])]
print(list(flatten_recursive(nested)))

def flatten_stack(seq):
    stack = [iter(seq)]
    while stack:
        for item in stack[-1]:
            if isinstance(item, (list, tuple)):
                stack.append(iter(item))
                break
            else:
                yield item
        else:
            stack.pop()
import time
start= time.time()
list(flatten_recursive(((((((((((((((((((((((((1,2,3))))))))))))))))))))))))))  
p = time.time() - start
print(p)
start = time.time()
list(flatten_stack(((((((((((((((((((((((((1,2,3))))))))))))))))))))))))))
list(flatten_stack(nested))
h = time.time()-start
print(h)
print(h>p, h/p)