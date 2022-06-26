# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural_256() :: 211152 μs, 779773779 Cycles
:: norm_neural_16()  :: 8214 μs, 30335301 Cycles
:: norm()            :: 304 μs, 1122468 Cycles
:: norm_inv()        :: 298 μs, 1102230 Cycles
:: norm_intrin()     :: 216 μs, 798423 Cycles

Accuracy Test
InvSqrt:   0.000961
Intrinsic: 0.000063
Neural16:  0.999504
Neural256: 1.784982
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually.

There seems to be an overfitting problem that the python `predict_x` dataset failed to detect? The C implementation of the network seems fine otherwise.
