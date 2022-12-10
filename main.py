import numpy as np
import emnist
import layers
import classifier

hidden_layers = [["conv", 2],
                 ["relu"],
                 ["pool"],
                 ["flat"],
                 ["line", 16]]

cnn = classifier.CNN([2, 10, 10], hidden_layers, 10)

# t_conv = layers.Convolution([2, 10, 10], 2)
# t_relu = layers.ReLU
# t_pool = layers.Pooling([2, 8, 8])
# t_flat = layers.Flatten
# t_line = layers.Linear([1, 32], 16)
# t_outp = layers.Output([1, 16], 10)

data1 = [[i+j for i in range(10)] for j in range(10)]

data2 = [[i*10+j*10 for i in range(10)] for j in range(10)]
data = np.array([data1, data2]) / 1000

print(cnn.predict(data))

# print("origin")
# print(data.shape)
# print(data)

# data = t_conv.forward(data)

# print("conv")
# print(data.shape)
# print(data)

# data = t_relu.forward(data)

# print("relu")
# print(data.shape)
# print(data)

# data = t_pool.forward(data)

# print("pool")
# print(data.shape)
# print(data)

# data = t_flat.forward(data)

# print("flat")
# print(data.shape)
# print(data)

# data = t_line.forward(data)

# print("line")
# print(data.shape)
# print(data)

# data = t_outp.forward(data)

# print("outp")
# print(data.shape)
# print(data)

# test = layers.Convolution([2, 10, 10], 2)

# test_pool = layers.Pooling([2, 10, 10])

# test_flat = layers.Flatten

# test_lin = layers.Linear([1, 10], 20)

# data1 = [[i+j for i in range(10)] for j in range(10)]

# data2 = [[i*10+j*10 for i in range(10)] for j in range(10)]

# data3 = [i for i in range(10)]
# data = np.array([data3])

# print(data.shape)
# print(data)

# result = test_lin.forward(data)
# print(result)
# print(result.shape)

# # print(test_flat.forward(data))
# # print(test_flat.forward(data).shape)

# # result = test_pool.forward(data)
# # print(result)
# # print(result.shape)


# # # print(test.W)

# # # print(test.im2col(data).shape)
# # # print(test.im2col(data))

# # result = test.forward(data)
# # print(result)
# # print(result.shape)

# # relu = layers.ReLU

# # print(relu.forward(result))