import numpy as np

class Convolution:
    def __init__(self, prev_layer_size, layer_size):
        self.prev_layer_size = prev_layer_size
        self.layer_size = layer_size
        self.filter_size = 3
        self.stride = 1
        self.out_h = (prev_layer_size[1] - self.filter_size) // self.stride + 1
        self.out_w = (prev_layer_size[2] - self.filter_size) // self.stride + 1
        
        # He初期化　重み：(レイヤーサイズ, チャンネル数, フィルターサイス, フィルターサイズ)
        self.W = 0.1 * np.random.randn(layer_size, prev_layer_size[0], self.filter_size, self.filter_size)
        self.b = np.zeros((layer_size))
        
    def forward(self, A_prev):
        A_prev_col = self.im2col(A_prev)
        filter_col = self.W.reshape(self.layer_size, -1).T
        
        Z_col = np.dot(A_prev_col, filter_col) + self.b
        Z = Z_col.reshape(self.layer_size, self.out_h, self.out_w, -1).transpose(0, 3, 1, 2)
        
        return Z[0]
        
    def im2col(self, A_prev):
        col = np.zeros((self.layer_size, self.prev_layer_size[0], self.filter_size, self.filter_size, self.out_h, self.out_w))
        print(col.shape)
        
        for y in range(self.filter_size):
            y_max = y + self.stride*self.out_h
            for x in range(self.filter_size):
                x_max = x + self.stride*self.out_w
                col[:, :, y, x, :, :] = A_prev[:, y:y_max:self.stride, x:x_max:self.stride]
                
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(self.layer_size * self.out_h * self.out_w, -1)
        return col        
    
class Linear:
    def __init__(self, prev_layer_size, layer_size):
        # He初期化
        self.W = 0.1*np.random.randn(prev_layer_size[1], layer_size)
        self.b = np.zeros((layer_size))
        
        print(self.W)
        
    def forward(self, A_prev):
        Z = np.dot(A_prev, self.W) + self.b
        A = self.ReLU(Z)
        
        return A
    
    def ReLU(self, Z):
        return np.maximum(Z, 0)
        
        
class Pooling:
    def __init__(self, prev_layer_size):
        self.prev_layer_size = prev_layer_size
        self.filter_size = 2
        self.stride = 2
        self.out_h = int(1 + (prev_layer_size[1] - self.filter_size) / self.stride)
        self.out_w = int(1 + (prev_layer_size[2] - self.filter_size) / self.stride)
        
    def forward(self, A_prev):
        A_prev_col = self.im2col(A_prev).reshape(-1, self.filter_size * self.filter_size)
        
        arg_max = np.argmax(A_prev_col, axis=1)
        Z_col = np.max(A_prev_col, axis=1)
        Z = Z_col.reshape(self.out_h, self.out_w, self.prev_layer_size[0]).transpose(2, 0, 1)
        
        return Z
        
        
    def im2col(self, A_prev):
        col = np.zeros((1, self.prev_layer_size[0], self.filter_size, self.filter_size, self.out_h, self.out_w))
        print(col.shape)
        
        for y in range(self.filter_size):
            y_max = y + self.stride*self.out_h
            for x in range(self.filter_size):
                x_max = x + self.stride*self.out_w
                col[:, :, y, x, :, :] = A_prev[:, y:y_max:self.stride, x:x_max:self.stride]
                
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(1 * self.out_h * self.out_w, -1)
        return col

class ReLU:
    def forward(A_prev):
        Z = np.maximum(A_prev, 0)
        return Z
    
class Flatten:
    def forward(A_prev):
        Z = A_prev.reshape(-1)
        return Z