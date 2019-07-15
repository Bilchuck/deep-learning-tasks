import datetime
import torch.nn.functional as F
from simple_conv_net_train import SimpleConvNet
from simple_conv_net_func import *

def expectGoodEnough(a, b):
    assert diff_mse(a, b) < 1e-6


device = 'cpu'
model = SimpleConvNet(device)
X = torch.rand([4,1,28,28])

conv_weight = model.conv_layer.weight
conv_bias = model.conv_layer.bias

weight = model.fc_layer1.weight
bias = model.fc_layer1.bias

print("Test vector implementation")
print("Conv 2d")
a = model.conv_layer(X)
b = conv2d_vector(X, conv_weight, conv_bias, device)
expectGoodEnough(a, b)
convolved = model.conv_layer(X)
print(f"passed!")

print("Pooling")
expectGoodEnough(F.max_pool2d(convolved, 2, 2), pool2d_vector(convolved, device))
pooled = F.max_pool2d(convolved, 2, 2)
print(f"passed!")

print("reshape")
expectGoodEnough(pooled.view(-1, 20*12*12), reshape_vector(pooled, device))
reshaped = pooled.view(-1, 20*12*12)
print(f"passed!")

print("fc layer")
expectGoodEnough(model.fc_layer1(reshaped), fc_layer_vector(reshaped, weight, bias, device))
fc_layered = model.fc_layer1(reshaped)
print(f"passed!")

print("relu")
expectGoodEnough(F.relu(fc_layered), relu_vector(fc_layered, device))
print(f"passed!")


print("Test scalar implementation:")
print("Conv 2d")
a = model.conv_layer(X)
b = conv2d_scalar(X, conv_weight, conv_bias, device)
expectGoodEnough(a, b)
convolved = model.conv_layer(X)
print(f"passed!")

print("Pooling")
expectGoodEnough(F.max_pool2d(convolved, 2, 2), pool2d_scalar(convolved, device))
pooled = F.max_pool2d(convolved, 2, 2)

print("reshape")
expectGoodEnough(pooled.view(-1, 20*12*12), reshape_scalar(pooled, device))
reshaped = pooled.view(-1, 20*12*12)
print(f"passed!")

print("fc layer")
expectGoodEnough(model.fc_layer1(reshaped), fc_layer_scalar(reshaped, weight, bias, device))
fc_layered = model.fc_layer1(reshaped)
print(f"passed!")

print("relu")
expectGoodEnough(F.relu(fc_layered), relu_scalar(fc_layered, device))
print(f"passed!")