# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can reproduce a vector normalisation function.

```
Speed Test
:: norm_neural_6x32() :: 434614 μs, 1171234377 Cycles
:: norm_neural_256()  :: 60934  μs, 164209167 Cycles
:: norm_neural_16()   :: 3135   μs, 8448840 Cycles
:: norm()             :: 276    μs, 741501 Cycles
:: norm_inv()         :: 284    μs, 765855 Cycles
:: norm_intrin()      :: 204    μs, 549585 Cycles

Accuracy Test
InvSqrt:    0.000961
Intrinsic:  0.000064
Neural16:   0.152058
Neural256:  0.038772
Neural6x32: 0.087609

```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

The 'Accuracy' value is the euclidean distance between the vector produced by the function and the actual vector produced by the accurate function which is a `1.f/sqrtf()` vector normalisation function.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually. An average accuracy of ~0.04 is actually really bad because all summed parts of a unit vector are to precisely add up to 1, so if the average accuracy is 0.04 off the original function, that's enough to have completely ruined our unit vector in most use cases.

The idea that a neural network can learn complex functions is honestly mostly a myth, or very rare/specific cases and even so they're not efficient at doing so, they really are just machines that seperate points of data into classes with shoddy interpolation _(by that I mean extrapolation)_ between the points - because activation functions are not designed to seperate data in a way that is necessarily interpolatable, it's just a side effect of the circumstances.
