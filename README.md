# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural_256() :: 201875 μs, 745523323 Cycles
:: norm_neural_16()  :: 6571 μs, 24263897 Cycles
:: norm()            :: 307 μs, 1132459 Cycles
:: norm_inv()        :: 303 μs, 1120952 Cycles
:: norm_intrin()     :: 218 μs, 805564 Cycles

Accuracy Test
InvSqrt:   0.000965
Intrinsic: 0.000063
Neural16:  0.905226
Neural256: 2.096257
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually.
