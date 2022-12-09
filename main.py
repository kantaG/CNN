import numpy as np
import emnist
import layers

test = layers.Convolution([2, 10, 10], 2)

test_pool = layers.Pooling([2, 10, 10])

test_flat = layers.Flatten

test_lin = layers.Linear([1, 10], 20)

data1 = [[i+j for i in range(10)] for j in range(10)]

data2 = [[i*10+j*10 for i in range(10)] for j in range(10)]

data3 = [i for i in range(10)]
data = np.array([data3])

print(data.shape)
print(data)

result = test_lin.forward(data)
print(result)
print(result.shape)

# print(test_flat.forward(data))
# print(test_flat.forward(data).shape)

# result = test_pool.forward(data)
# print(result)
# print(result.shape)


# # print(test.W)

# # print(test.im2col(data).shape)
# # print(test.im2col(data))

# result = test.forward(data)
# print(result)
# print(result.shape)

# relu = layers.ReLU

# print(relu.forward(result))