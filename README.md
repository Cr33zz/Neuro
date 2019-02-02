# Neuro
C# implementation of neural networks library. Contains basic types of layers (dense, convolution, pooling, flatten). Supports single CPU, Multi-CPU and GPU tensor operations (using CUDAfy).

Sample sequential network

```
var net = new NeuralNetwork("deep_dense_test");
var model = new Sequential();
model.AddLayer(new Dense(inputs, 5, Activation.Linear));
model.AddLayer(new Dense(net.LastLayer, 4, Activation.Linear));
model.AddLayer(new Dense(net.LastLayer, inputs, Activation.Linear));
net.Model = model;
net.Optimize(new SGD(0.02f), Loss.MeanSquareError);

List<Data> tData = new List<Data>();
for (int i = 0; i < 100; ++i)
{
    var input = new Tensor(net.Layer(0).InputShape);
    input.FillWithRand();
    tData.Add(new Data() { Input = input, Output = input.Mul(1.7f) });
}

net.Fit(tData, 10, 50, null, 2, Track.TrainError);
```

Sample flow network (streams)

```
var net = new NeuralNetwork("test");

LayerBase mainInput = new Dense(2, 2, Activation.Linear) { Name = "main_input" };
LayerBase auxInput = new Input(new Shape(1, 2)) { Name = "aux_input" };
LayerBase concat = new Concatenate(new []{ mainInput, auxInput }) { Name = "concat" };

net.Model = new Flow(new[] { mainInput, auxInput }, new[] { concat });
net.Optimize(new SGD(0.05f), Loss.MeanSquareError);

var inputs = new[] { new Tensor(new float[] { 0, 1 }, new Shape(1, 2)),
                     new Tensor(new float[] { 1, 2 }, new Shape(1, 2)) };
var output = new Tensor(new float[] { 1, 2, 1, 2 }, new Shape(1, 4));
var trainingData = new List<Data> { new Data(inputs, new []{output}) };

net.Fit(trainingData, 1, 50, null, 0, Track.Nothing, false);
```
