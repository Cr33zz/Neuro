# Neuro
C# implementation of neural networks library. Contains basic types of layers (dense, convolution, pooling, flatten). Supports single CPU, Multi-CPU and GPU tensor operations (using CUDAfy).

Sample test network

```
var net = new NeuralNetwork("deep_dense_test");
net.AddLayer(new Dense(inputs, 5, Activation.Linear));
net.AddLayer(new Dense(net.LastLayer, 4, Activation.Linear));
net.AddLayer(new Dense(net.LastLayer, inputs, Activation.Linear));

List<Data> tData = new List<Data>();
for (int i = 0; i < 100; ++i)
{
    var input = new Tensor(net.Layer(0).InputShape);
    input.FillWithRand();
    tData.Add(new Data() { Input = input, Output = input.Mul(1.7f) });
}

net.Optimize(new SGD(0.02f), Loss.MeanSquareError);
net.Fit(tData, batchSize, epochs, null, 2, Track.TrainError);
```
