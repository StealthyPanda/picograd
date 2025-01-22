from typing import Any, Callable
import numpy as np


_def_dtype = np.float32

# Simple operations
# Note: Each operation has 2 backward functions that take the current forward values of operands(x and y) and the 
# value of the gradient of their resultant (z) and return the gradient values for x (1st function) and y (2nd function)

# z = x + y
_add = (
    lambda x, y, z : z,
    lambda x, y, z : z,
)

# z = x - y
_sub = (
    lambda x, y, z :  z,
    lambda x, y, z : -z,
)

# z = x * y
_mul = (
    lambda x, y, z : z * y,
    lambda x, y, z : z * x,
)

# z = x / y
_div = (
    lambda x, y, z :  z / y,
    lambda x, y, z : -z * x/(y**2),
)

# z = x ** y
_pow = (
    lambda x, y, z :  z * y * (x ** (y - 1)),
    lambda x, y, z : z * np.log(x) * (x ** y),
)




def _mat_x(x : np.ndarray, y : np.ndarray, z : np.ndarray) -> np.ndarray:
    '''
    returns the gradient for x, given z = x @ y
    '''
    trans = y
    if len(y.shape) >= 2:
        perm = list(range(len(y.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        trans = np.transpose(y, perm)
    elif len(y.shape) == 0:
        trans = np.expand_dims(y, 0)
    
    return z @ trans if len(z.shape) > 0 else z.item() * trans

def _mat_y(x : np.ndarray, y : np.ndarray, z : np.ndarray) -> np.ndarray:
    '''
    returns the gradient for y, given z = x @ y
    '''
    trans = x
    if len(x.shape) >= 2:
        perm = list(range(len(x.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        trans = np.transpose(x, perm)
    elif len(x.shape) == 0:
        trans = np.expand_dims(x, 0)
    
    return trans @ z if len(z.shape) > 0 else trans * z.item()

# z = x @ y
_mat = (_mat_x, _mat_y)



class tensor:
    '''Base class for tensors. The underlying values are stored in numpy arrays.'''
    
    def __init__(
            self, value : int | float | np.ndarray = 0,
            rev_op : Callable[..., object] | tuple[Callable[..., object], Callable[..., object]] = None, 
            args : list = None, grad : bool = False
        ) -> None:
        """Creates a tensor.

        Args:
            value (int | float | np.ndarray, optional): Value of this tensor. Can be any iterable. Defaults to 0.
            
            rev_op (Callable[..., object] | tuple[Callable[..., object], Callable[..., object]], optional): Reverse operation to apply duing reverse mode autodiff. You shouldn't mess with this.
            
            args (list, optional): Args for the reverse operation. Don't mess with this either. Defaults to None.
            
            grad (bool, optional): Whether this tensor should be tracked for a gradient. You want to set this to true for model weights etc. Defaults to False.
        """
        
        #Value of this tensor (numpy array)
        self.value = np.array(value, dtype = _def_dtype)
        
        #size of this tensor (number of individual elements)
        self.size = np.size(self.value)
        
        #gradient of this tensor. could be none.
        self.grad = np.zeros_like(self.value, dtype = _def_dtype) if grad else None
        
        #shape of this tensor.
        self.shape = self.value.shape
        
        #reverse operations associated with this tensor. could be none.
        self.rev_op = rev_op
        
        #arguments / children of this tensor, if it the result of an operatin. could be [].
        self.children : list[tensor] = args if args is not None else []
    
    def __repr__(self) -> str:
        return f'tensor{"(tracked)" if self.has_grad else ""}<{self.shape}>'
    
    def __getitem__(self, key) -> np.ndarray:
        return (self.value.__getitem__(key))
    
    def __setitem__(self, key, value):
        self.value.__setitem__(key, value)
    
    @property
    def isroot(self) -> bool:
        '''returns if this tensor is a root (not a result of an operation)'''
        return self.rev_op is None
    
    @property
    def isintermediate(self) -> bool:
        '''returns if this tensor is the result of an operation (intermediate variable)'''
        return self.rev_op is not None
    
    @property
    def has_grad(self) -> bool:
        '''returns if this variable is being tracked for a gradient'''
        return self.grad is not None
    
    def track(self) -> object:
        '''
        Starts tracking this variable for a gradient. You want to call this (if not already set while creating this tensor)
        for model weights etc. Returns this tensor back.
        '''
        if self.grad is None:
            self.grad = np.zeros_like(self.value, dtype=_def_dtype)
        return self
    
    def untrack(self) -> object:
        '''Stops tracking this variable for gradient.'''
        self.grad = None
        return self
    
    
    
    
    
    
    def __add__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        return tensor(
            value = self.value + other.value, rev_op = _add, args = [self, other],
            grad = self.has_grad or other.has_grad
        )
    def __radd__(self, other : object) -> object:
        return tensor.__add__(other, self)
    
    def __sub__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        return tensor(
            value = self.value - other.value, rev_op = _sub, args = [self, other],
            grad = self.has_grad or other.has_grad
        )
    def __rsub__(self, other : object) -> object:
        return tensor.__sub__(other, self)
    
    
    
    def __truediv__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        return tensor(
            value = self.value / other.value, rev_op = _div, args = [self, other],
            grad = self.has_grad or other.has_grad
        )
    def __rtruediv__(self, other : object) -> object:
        return tensor.__truediv__(other, self)
    
    def __mul__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        return tensor(
            value = self.value * other.value, rev_op = _mul, args = [self, other],
            grad = self.has_grad or other.has_grad
        )
    def __rmul__(self, other : object) -> object:
        return tensor.__mul__(other, self)
    
    def __neg__(self) -> object:
        return tensor(
            value = -self.value, rev_op=(lambda x, z: -z, ), args = [self],
            grad = self.has_grad
        )
    
    def __matmul__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        return tensor(
            value = self.value @ other.value, rev_op = _mat, args = [self, other],
            grad = self.has_grad or other.has_grad
        )
    def __rmatmul__(self, other : object) -> object:
        return tensor.__matmul__(other, self)
    
    def __pow__(self, other : object) -> object:
        other = to_tensor(other)
        self = to_tensor(self)
        try:
            return tensor(
                value = self.value ** other.value, rev_op = _pow, args = [self, other],
                grad = self.has_grad or other.has_grad
            )
        except FloatingPointError as e:
            print(f'Overflow, {self.shape} and {other.value}')
            print(e)
    def __rpow__(self, other : object) -> object:
        return tensor.__pow__(other, self)
    
    @property
    def item(self) -> float:
        '''Returns the single item in the tensor, if the size of the tensor is 1.'''
        return self.value.item()
    
    def flow_back(self) -> None:
        '''
        Don't call this function directly unless you know what you're doing.
        Flows the gradient backwards recursively. Use .reverse() for actually finding the gradients.
        '''
        if self.isintermediate and self.has_grad:
            xandy = [x.value for x in self.children]
            for i, each in enumerate(self.children):
                if not each.has_grad: continue
                change = self.rev_op[i](*xandy, self.grad)
                try : each.grad += change
                except ValueError:
                    print(f'Skipping flow for {each.grad.shape} with {change.shape}...')
                
                if each.isintermediate and each.has_grad: each.flow_back()
    
    
    def reverse(self) -> None:
        '''Flows gradient backwards from this variable to all the previous tracked ones in the DAG.'''
        self.grad = np.ones_like(self.value, dtype = _def_dtype)
        self.flow_back()
    
    
    def clear_grads(self) -> None:
        '''Sets all the gradients to 0 in the DAG.'''
        if not self.has_grad : return
        self.grad = np.zeros_like(self.grad)
        for each in self.children: each.clear_grads()
    
    def clone(self) -> None:
        '''Creates a copy of this tensor, which won't be a part of the original DAG.'''
        copy = tensor(value = self.value.copy())
        copy.grad = self.grad.copy()
        return copy


def to_tensor(value : Any) -> tensor:
    '''Converts any iterable, or scalar value to a tensor. Must be numeric.'''
    if type(value) == tensor: return value
    return tensor(value = value)


def rand(*shape : int, sigma : float = 0.05, mu : float = 0) -> tensor:
    '''Returns `shape` shaped tensor of random numbers sampled from a normal distribution.'''
    return tensor(mu + (sigma * np.random.randn(*shape)))

def glorot_uniform(*shape : int, in_size : int, out_size : int) -> tensor:
    '''Returns `shape` shaped tensor of random numbers sampled from a uniform glorot distribution.'''
    high = (6 / (in_size + out_size)) ** 0.5
    low = -high
    return tensor(np.random.uniform(low=low, high=high, size=shape))

def glorot_normal(*shape : int, in_size : int, out_size : int) -> tensor:
    '''Returns `shape` shaped tensor of random numbers sampled from a normal glorot distribution.'''
    sigma = (2 / (in_size + out_size)) ** 0.5
    return tensor(np.random.randn(*shape) * sigma)


def zeros(*shape : int) -> tensor:
    '''Returns `shape` shaped 0s tensor.'''
    return tensor(np.zeros(shape, dtype=_def_dtype))

def zeros_like(t : np.ndarray | list | tuple) -> tensor:
    '''Returns `t` shaped 0s tensor.'''
    return tensor(np.zeros_like(t, dtype=_def_dtype))

def ones(*shape : int) -> tensor:
    '''Returns `shape` shaped 1s tensor.'''
    return tensor(np.ones(shape, dtype=_def_dtype))

def ones_like(t : np.ndarray | list | tuple) -> tensor:
    '''Returns `t` shaped 1s tensor.'''
    return tensor(np.ones_like(t, dtype=_def_dtype))

def identity(n : int) -> tensor:
    '''Returns `(n, n)` shaped identity matrix.'''
    I = np.zeros((n, n), dtype = _def_dtype)
    for each in range(n):
        I[each, each] = 1
    return tensor(value = I)


def sin(x : tensor) -> tensor:
    '''Returns element-wise sin of this `x`'''
    z = tensor(value = np.sin(x.value), rev_op = (lambda t, z: z * np.cos(t),), args = [x], grad=x.has_grad)
    return z

def cos(x : tensor) -> tensor:
    '''Returns element-wise cos of this `x`'''
    z = tensor(value = np.cos(x.value), rev_op = (lambda t, z: -z * np.sin(t),), args = [x], grad=x.has_grad)
    return z

def exp(x : tensor) -> tensor:
    '''Returns element-wise `e^x` of this `x`'''
    z = tensor(value = np.exp(x.value), rev_op = (lambda t, z: z * np.exp(t),), args = [x], grad=x.has_grad)
    return z

def sigmoid(x : tensor) -> tensor:
    '''Returns element-wise sigmoid of this `x`'''
    return (1 / (1 + exp(-x)))

def sum(x : tensor, axis : int = -1) -> tensor:
    '''Returns sum of `x` along `axis`. `axis` can be any integer, as long as it is a valid dimension.
    Defaults to -1.'''
    return x @ ones(x.shape[axis], 1) if axis == -1 else ones(1, x.shape[axis]) @ x


def tile(x : tensor, n : int, axis : int = 0) -> tensor:
    '''Tiles `x` along the at the given axis position, adding that new dimension.
    For eg. if `x.shape = [d1, d2, ...]` then `tile(x, n).shape = [n, d1, d2, ...]`. Axis is 0 by default.
    '''
    tiles = [1 for _ in range(len(x.shape) + 1)]
    tiles[axis] = n
    return tensor(
        value = np.tile(np.expand_dims(x.value, axis), tiles),
        rev_op = (lambda t, z : (np.sum(z, axis = axis) / n), ),
        args = [x], grad=x.has_grad
    )


def squeeze(x : tensor, axis : int = -1) -> tensor:
    '''
    Removes a dimension whose length is 1. For eg. if `x.shape = [1, 2, 1, 5]` then 
    `squeeze(x, 2).shape = [1, 2, 5]`. Axis defaults to -1.
    '''
    return tensor(
        value = np.squeeze(x.value, axis),
        rev_op = (lambda t, z : np.expand_dims(z, axis), ),
        args = [x], grad=x.has_grad
    )

def unsqueeze(x : tensor, axis : int = -1) -> tensor:
    '''
    Add a dimension whose length is 1. For eg. if `x.shape = [1, 2, 5]` then 
    `unsqueeze(x, 2).shape = [1, 2, 1, 5]`. Axis defaults to -1.
    '''
    return tensor(
        value = np.expand_dims(x.value, axis),
        rev_op = (lambda t, z : np.squeeze(z, axis), ),
        args = [x], grad=x.has_grad
    )

def expand(x : tensor, n : int, axis : int = -1) -> tensor:
    '''
    Expands a dimension whose length is 1. For eg. if `x.shape = [1, 2, 1, 5]` then 
    `expand(x, 15, 2).shape = [1, 2, 15, 5]`. Axis defaults to -1.
    '''
    reps = [1 for _ in x.shape]
    reps[axis] = n
    return tensor(
        value = np.tile(x.value, reps),
        rev_op = (lambda t, z : np.expand_dims(np.sum(z, axis) / n, axis=axis), ),
        args = [x], grad=x.has_grad
    )


def softmax(x : tensor, axis : int) -> tensor:
    '''
    Returns the softmax of this tensor.
    '''
    x = exp(x)
    y = sum(x, axis=axis)
    y = expand(y, x.shape[axis], axis)
    x = x / y
    return x

def _relu_dash(x : np.ndarray, threshold : float = 0) -> np.ndarray:
    y = x.copy()
    y[x >= threshold] = 1
    y[x <  threshold] = 0
    return y

def relu(x : tensor, threshold : float = 0) -> tensor:
    '''Returns element-wise ReLU; `ReLU(x - threshold)`'''
    nxv = x.value.copy()
    nxv[nxv < threshold] = 0
    return tensor(
        nxv,
        (lambda t, z : z * _relu_dash(t, threshold), ),
        args = [x], grad = x.has_grad
    )

def reshape(x : tensor, new_shape : list[int] | tuple[int]) -> tensor:
    '''Reshapes the given tensor to new shape. This doesn't change the actual arrangement of data, so don't 
    worry about it being expensive in terms of memory or anything.'''
    original_shape = x.shape
    return tensor(
        x.value.reshape(new_shape),
        rev_op = (lambda t, z: z.reshape(original_shape), ),
        args=[x], grad=x.has_grad
    )





