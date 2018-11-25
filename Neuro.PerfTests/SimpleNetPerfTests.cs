using Neuro.Layers;
using Neuro.Tensors;
using Neuro.Optimizers;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class SimpleNetPerfTests
    {
        static void Main(string[] args)
        {
            var inputs = new Tensor(new double[] { 1,1,2,2,3,3,4,4,5,5,6,6 }, new Shape(1, 2, 1, 6));
            var outputs = new Tensor(new double[] { 2,2,3,3,4,4,5,5,6,6,7,7 }, new Shape(1, 2, 1, 6));

            var net = new NeuralNetwork("test");
            net.AddLayer(new Dense(2, 5, Activation.Sigmoid));
            net.AddLayer(new Dense(net.LastLayer, 4, Activation.Sigmoid));
            net.AddLayer(new Dense(net.LastLayer, 2, Activation.Linear));
            net.Optimize(new Adam(0.01), Loss.MeanSquareError);

            var l0 = net.Layer(0) as Dense;
            l0.Weights = new Tensor(new[] {-0.5790837 ,  0.79525125, -0.6933877 , -0.3692013 ,  0.1810553,
                                            0.03039712,  0.91264546,  0.11529088,  0.33134186, -0.46221718 }, new Shape(l0.Weights.Height, l0.Weights.Width)).Transposed();

            var l1 = net.Layer(1) as Dense;
            l1.Weights = new Tensor(new[] { 0.08085728, -0.10262775,  0.38443696, -0.23273587,
                                            0.33498216, -0.7566199 , -0.814561  , -0.08565235,
                                           -0.55490625,  0.6140275 ,  0.34785295, -0.3431782,
                                            0.47427893, -0.41688982,  0.59143007,  0.00616223,
                                            0.60304165,  0.6548513 , -0.78456855,  0.4640578 }, new Shape(l1.Weights.Height, l1.Weights.Width)).Transposed();

            var l2 = net.Layer(2) as Dense;
            l2.Weights = new Tensor(new[] { 0.32492328,  0.6930735,
                                           -0.7263415 ,  0.4574399,
                                            0.5422747 ,  0.19008946,
                                            0.911242  , -0.24971604 }, new Shape(l2.Weights.Height, l2.Weights.Width)).Transposed();

            Trace.WriteLine(net.Predict(inputs.GetBatch(0)));

            net.Fit(inputs, outputs, -1, 10, 2, Track.Nothing, false);

            /*var inShape = new Shape(20);
            var outShape = new Shape(20);

            List<Data> trainingData = new List<Data>();

            for (int i = 0; i < 500; ++i)
            {
                var input = new Tensor(inShape);
                input.FillWithRand(3 * i);
                var output = new Tensor(outShape);
                output.FillWithRand(3 * i);
                trainingData.Add(new Data() { Input = input, Output = output });
            }

            var net = new NeuralNetwork("simple_net_perf_test");
            net.AddLayer(new Flatten(inShape));
            net.AddLayer(new Dense(net.LastLayer, 24, Activation.ReLU));
            net.AddLayer(new Dense(net.LastLayer, 24, Activation.ReLU));
            net.AddLayer(new Dense(net.LastLayer, outShape.Length, Activation.Linear));
            net.Optimize(new Adam(), Loss.MeanSquareError);

            var timer = new Stopwatch();
            timer.Start();

            net.Fit(trainingData, 1, 100, null, 0, Track.Nothing);

            timer.Stop();
            Console.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");*/

            return;
        }
    }
}
