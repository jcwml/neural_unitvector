# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural() :: 5822650 μs, 21502816659 Cycles
:: norm()        :: 3084593 μs, 11391277505 Cycles
:: norm_inv()    :: 3078738 μs, 11369655519 Cycles
:: norm_intrin() :: 2210105 μs, 8161826041 Cycles

Accuracy Test
InvSqrt: 0.034
Intrinsic: 0.034
Neural: 1125899.875
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.
