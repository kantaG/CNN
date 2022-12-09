import numpy as np
import emnist
import layers

test = layers.Convolution([2, 10, 10], 1)

data1 = [[i+j for i in range(10)] for j in range(10)]

data2 = [[i*10+j*10 for i in range(10)] for j in range(10)]
data = np.array([data1, data2])

print(data.shape)
print(data)

# print(test.W)

print(test.im2col(data).shape)
print(test.im2col(data))