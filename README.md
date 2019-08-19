# fashion-mnist-pytorch

The a simple CNN to classify [fashion mnist][1]. The following network provides 90% test accuracy.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             832
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
         MaxPool2d-4           [-1, 32, 13, 13]               0
            Conv2d-5           [-1, 64, 13, 13]          18,496
              ReLU-6           [-1, 64, 13, 13]               0
       BatchNorm2d-7           [-1, 64, 13, 13]             128
         MaxPool2d-8             [-1, 64, 6, 6]               0
            Linear-9                 [-1, 1024]       2,360,320
             ReLU-10                 [-1, 1024]               0
           Linear-11                   [-1, 10]          10,250
================================================================
Total params: 2,390,090
Trainable params: 2,390,090
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.82
Params size (MB): 9.12
Estimated Total Size (MB): 9.94
----------------------------------------------------------------
```

[1]: https://github.com/zalandoresearch/fashion-mnist
