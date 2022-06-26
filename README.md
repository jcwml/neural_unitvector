# neural_unitvector
A Feed-forward Neural Network trained to learn a vector normalisation function.

This will generate x number of random points in three dimensional space for the x dataset and normalise them for the y dataset, the end result is a network that can reproduce a vector normalisation function.

```
Speed Test
:: norm_neural_6x32() :: 479753 μs, 1771679954 Cycles
:: norm_neural_256()  :: 194167 μs, 717036394 Cycles
:: norm_neural_16()   :: 8562 μs, 31616944 Cycles
:: norm()             :: 310 μs, 1144484 Cycles
:: norm_inv()         :: 305 μs, 1124504 Cycles
:: norm_intrin()      :: 220 μs, 812890 Cycles

Accuracy Test
InvSqrt:    0.000964
Intrinsic:  0.000063
Neural16:   1.021691
Neural256:  0.945880
Neural6x32: 0.926647
```

As you can see, the neural network is the least performing. This test used a smaller neural network with an accuracy of 0.90 for efficient computation over accuracy however even with auto vectorisation for fma enabled the neural version still came out significantly slower.

The 'Accuracy' value is the euclidean distance between the vector produced by the function and the actual vector produced by the accurate function which is a `1.f/sqrtf()` vector normalisation function.

It is unlikely that a neural unit vector function will ever compete with the traditional normalisation functions in speed or accuracy combined or individually. An average accuracy of ~1 is actually really bad because all summed parts of a unit vector are to add up to 1, so if the average accuracy is 1 over the original function, that's enough to have completely ruined our unit vector.

There is an overfitting problem when training in Python, as such I now use datasets generated in the C program by uncommenting the code [here](https://github.com/jcwml/neural_unitvector/blob/main/main.c#L236) and import the generated training and test sets into Python, this gives a case that produces a better trained network and less overfitted results, although still somewhat overfitted in python. At the end of the day teaching a neural network of this nature to generate a vector normalisation function without overfitting to the dataset is going to be near impossible, neural networks just seperate data for classification purposes and at best they can somewhat interpolate between these datapoints in a fashion that is somewhat bent depending on the nonlinearity of the activation function you are using but this bending or "warping" of the interpolation between points is less noticable at higher sampled datasets.

The idea that a neural network can learn complex functions is honestly mostly a myth, or very rare/specific cases and even so they're not efficient at doing so, they really are just machines that seperate points of data into classes with shoddy interpolation between the points because activation functions are not designed to seperate data in a way that is necessarily interpolatable, it's just a side effect of the circumstances.

Also neural networks scale REALLY BADLY when it comes to getting higher accuracy results, you can see the amount of added complexity to these neural networks for VERY MINIMAL gains is ridiculous. Excuse my use of upper case.

Context:
```
Speed Test
:: norm_neural_6x32() :: 479753 μs, 1771679954 Cycles
:: norm_neural_256()  :: 194167 μs, 717036394 Cycles
:: norm_neural_16()   :: 8562 μs, 31616944 Cycles

Accuracy Test
Neural16:   1.021691
Neural256:  0.945880
Neural6x32: 0.926647
```
Neural256 is 0.075811 more accurate than Neural16 for 185,605 μs more compute time that's 22.6x more compute time for ~1/13th more accuracy.

It seems fair to say the gains get exponentially worse in terms of "bang for buck".
