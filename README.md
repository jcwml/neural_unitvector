# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural_256() :: 222270 μs, 820833382 Cycles
:: norm_neural_16()  :: 8586 μs, 31707113 Cycles
:: norm()            :: 311 μs, 1149997 Cycles
:: norm_inv()        :: 305 μs, 1129314 Cycles
:: norm_intrin()     :: 221 μs, 817367 Cycles

Accuracy Test
InvSqrt:   0.000965
Intrinsic: 0.000063
Neural16:  1.022055
Neural256: 0.944933
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

The 'Accuracy' value is the euclidean distance between the vector produced by the function and the actual vector produced by the accurate function which is a `1.f/sqrtf()` vector normalisation function.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually. An average accuracy of ~1 is actually really bad because all summed parts of a unit vector are to add up to 1, so if the average accuracy is 1 over the original function, that's enough to have completely ruined our unit vector.

There seems to be an overfitting problem that the python `predict_x` dataset failed to detect? The C implementation of the network seems fine otherwise _(I was right)_.
