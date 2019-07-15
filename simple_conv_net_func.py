from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.contiguous().view(1, -1).squeeze()
    y_vec = y.contiguous().view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    (N, C_in, S_in, _) = x_in.shape
    (C_out, _, K, _) = conv_weight.shape
    
    S_out = S_in - K + 1
    
    Z = torch.empty(N, C_out, S_out, S_out).to(device)
    
    for n_i in range(N):
        for c_out in range(C_out):
            for y in range(S_out):
                for x in range(S_out):
                    Z[n_i, c_out, y, x] = 0
                    for c_in in range(C_in):
                        for ky in range(K):
                            for kx in range(K):
                                Z[n_i, c_out, y, x] += conv_weight[c_out, c_in, ky, kx] * x_in[n_i, c_in, y+ky, x+kx]
                    Z[n_i, c_out, y, x] += conv_bias[c_out]
    return Z


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    N, C_in, S_in, _ = x_in.shape
    C_out, _, K, _ = conv_weight.shape

    S_out = S_in - K + 1

    X_col = im2col(x_in, K, device)
    conv_weight_rows = conv_weight2rows(conv_weight)

    Z = conv_weight_rows.mm(X_col).add(conv_bias.view((C_out, 1)))

    Z = Z.view((C_out, N, C_in, S_out, S_out)).sum(dim=2).transpose(0, 1)
    return Z


def im2col(X, kernel_size, device, stride=1):
    N_in, C_in, S_in, _ = X.shape
    S_out = (S_in - kernel_size) // stride + 1
    Z = torch.zeros((kernel_size, kernel_size, N_in, C_in, S_out, S_out)).to(device)

    for y in range(kernel_size):
        for x in range(kernel_size):
            y_to = y + stride * out_height
            x_to = x + stride * out_width
            Z[y, x, :, :, :, :] = X[:, :, y:y_to:stride, x:x_to:stride]
    return Z.view((kernel_size ** 2, -1))


def conv_weight2rows(conv_weight):
    conv_weight = conv_weight.clone()
    C_out, C_in = conv_weight.shape[0:2]
    return conv_weight.view((C_out * C_in, -1))


def pool2d_scalar(a, device):
    (N_in, C_in, S_in, _) = a.shape
    S_out = S_in // 2
    Z = torch.empty(N_in, C_in, S_out, S_out).to(device)
    
    for n_i in range(N_in):
        for c_in in range(C_in):
            for y in range(S_out):
                for x in range(S_out):
                    Z[n_i, c_in, y, x] = max([
                        a[n_i, c_in, y*2,   x*2    ],
                        a[n_i, c_in, y*2,   x*2+1  ],
                        a[n_i, c_in, y*2+1, x*2    ],
                        a[n_i, c_in, y*2+1, x*2 + 1],
                    ])
    return Z


def pool2d_vector(a, device):
    N, C_in, S_in, _ = a.shape
    S_out = S_in // 2

    return im2col(a, 2, device, stride=2).max(dim=0)[0].view(N, C_in, S_out, S_out)


def relu_scalar(a, device):
    N, length = X.shape
    Z = torch.empty(N, length).to(device)
    
    for N_in in range(N):
        for x in range(length):
            Z[N_in, x] = max(X[N_in, x], 0)
    return Z


def relu_vector(a, device):
    result = a.clone()
    result[a < 0] = 0
    return result


def reshape_vector(a, device):
    return a.clone().view((a.shape[0], -1))


def reshape_scalar(a, device):
    N, C_in, S_in, _ = a.shape
    Z = torch.empty((N, C_in * S_in * S_in)).to(device)

    for n_i in range(N):
        for c_in in range(C_in):
            for y in range(S_in):
                for x in range(S_in):
                    index = c_in * S_in * S_in + y * S_in + x
                    Z[n_i, index] = a[n_i, c_in, y, x]
    return Z

def fc_layer_scalar(a, weight, bias, device):
    N, length = a.shape
    length_out, _ = weight.shape
    
    Z = torch.empty(N, length_out).to(device)
    
    for N_in in range(N):
        for out_i in range(length_out):
            Z[N_in, out_i] = 0
            for in_i in range(length):
                Z[N_in, out_i] += a[N_in, in_i] * weight[out_i, in_i]
            Z[N_in, out_i] +=bias[out_i]
    return Z


def fc_layer_vector(a, weight, bias, device):
    return a.mm(weight.t()).add(bias)
