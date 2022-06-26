# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural_256() :: 5835027 μs, 21548540001 Cycles
:: norm_neural_16()  :: 5953079 μs, 21984501048 Cycles
:: norm()            :: 3212882 μs, 11865056104 Cycles
:: norm_inv()        :: 3304120 μs, 12201993533 Cycles
:: norm_intrin()     :: 2320295 μs, 8568761550 Cycles

Accuracy Test
InvSqrt:   0.033554
Intrinsic: 0.033554
Neural16:  1125899.875000
Neural256: 562949.937500
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

There seems to be no significant performance loss over using the larger neural network over the smaller one.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually.
