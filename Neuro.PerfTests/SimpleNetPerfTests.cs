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
            var inShape = new Shape(20);
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
            Console.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");

            return;
        }
    }
}
