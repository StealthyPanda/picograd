<center>
<h1><center>picograd</center></h1>
<!-- <h3>yet another autodiff engine</h3> -->
<center>Yet another autodiff engine</center>

---
</center>

![image](./image.png)


This is just another autodiff engine in python. There are many like it, but this one is mine.


On a real note, inspired by the infamous [micrograd]() and the countless stripped-down versions of it, I wanted to make the smallest possible library for autodiff. This one is different because it __supports tensors and its operations natively__, rather than working on scalars alone.

In terms of performance, of all the autodiff libraries out there, this is definitely one of them.

The whole thing is in one file, [pico.py](./pico.py) ~ 300 lines, most of which is comments and helper functions. The actual `tensor` class is ~ 80 lines in total, including comments.

The library is incredibly lightweight, easy to understand, and has enough as a base to extend it yourself for your own use. Go forth and extend!


### Usage
---
The code in pico.py is quite simple and self-explanatory. It has a small surface area, and is very similar to pytorch in usage.

Similar to literally any other library, the basic data type in pico is tensor:

```python
x = tensor([[1, 2, 3]])

print(x) # tensor<(1, 3)>
```

To track tensors for gradients, use `grad = true` or `.track()`

```python

y = tensor([
    [1],
    [3],
    [5],
], grad = True)

#equivalent to 
y = tensor([
    [1],
    [3],
    [5],
]).track()
```

To find gradients, use `.reverse()`:
```python
z = x @ y # 22
z.reverse()

print(x.grad) # [1, 3, 5]
print(y.grad) # [[1], [2], [3]]
```


A simple demo of workflow in finding gradients:
```python

from pico import *

#simple tensors:
c = tensor(grad = True)

#normally generated:
x = gaussian(5, 7).track()
y = gaussian(7, 9).track()


# z is now automatically tracked
z = (x @ y) #(5, 9)
ctile = tile(tile(c, 9), 5) # () -> (5, 9)
z = z + ctile

#calculate some loss
target = gaussian(5, 9)
loss = ((target - z) ** 2) / (5 * 9)

#flow the gradients back
loss.reverse()


print(x.grad)
print()
print(y.grad)
print()
print(c.grad)
```

---
_red panda image from [wikipedia](https://en.wikipedia.org/wiki/Red_panda)_
