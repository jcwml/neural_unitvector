# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can easily reproduce a fairly accurate vector normalisation function.

```
Speed Test
:: norm_neural_256()  :: 214403 μs, 791795227 Cycles
:: norm_neural_6x32() :: 462741 μs, 1708918852 Cycles
:: norm_neural_16()   :: 8751 μs, 32315911 Cycles
:: norm()             :: 399 μs, 1470306 Cycles
:: norm_inv()         :: 432 μs, 1594737 Cycles
:: norm_intrin()      :: 293 μs, 1082286 Cycles

Accuracy Test
InvSqrt:    0.000964
Intrinsic:  0.000063
Neural16:   1.021811
Neural256:  0.945282
Neural6x32: 0.925680
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

The 'Accuracy' value is the euclidean distance between the vector produced by the function and the actual vector produced by the accurate function which is a `1.f/sqrtf()` vector normalisation function.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually. An average accuracy of ~1 is actually really bad because all summed parts of a unit vector are to add up to 1, so if the average accuracy is 1 over the original function, that's enough to have completely ruined our unit vector.

There seems to be an overfitting problem that the python `predict_x` dataset failed to detect? The C implementation of the network seems fine otherwise _(I was right)_.

There is an overfitting problem when training in Python, as such I now use datasets generated in the C program by uncommenting the code [here](https://github.com/jcwml/neural_unitvector/blob/main/main.c#L236) and import the generated training and test sets into Python, this gives a case that produces a better trained network and less overfitted results, although still somewhat overfitted in python. At the end of the day teaching a neural network of this nature to generate a vector normalisation function without overfitting to the dataset is going to be near impossible, neural networks just seperate data for classification purposes and at best they can somewhat interpolate between these datapoints in a fashion that is somewhat bent depending on the nonlinearity of the activation function you are using but this bending or "warping" of the interpolation between points is less noticable at higher sampled datasets.

The idea that a neural network can learn complex functions is honestly mostly a myth, or very rare/specific cases and even so they're not efficient at doing so, they really are just machines that seperate points of data into classes with shoddy interpolation between the points because activation functions are not designed to seperate data in a way that is necessarily interpolatable, it's just a side effect of the circumstances.
