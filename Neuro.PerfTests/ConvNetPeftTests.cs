using System;
using System.Diagnostics;
using Neuro.Layers;
using Neuro.Tensors;
using Neuro.Optimizers;
using Neuro.Models;

namespace Neuro.PerfTests
{
    class ConvNetPeftTests
    {
        static void Main(string[] args)
        {
            Tensor.SetOpMode(Tensor.OpMode.GPU);

            var net = new NeuralNetwork("test");
            var inputShape = new Shape(64, 64, 4);
            var model = new Sequential();
            model.AddLayer(new Convolution(inputShape, 8, 32, 2, Activation.ELU));
            model.AddLayer(new Convolution(model.LastLayer, 4, 64, 2, Activation.ELU));
            model.AddLayer(new Convolution(model.LastLayer, 4, 128, 2, Activation.ELU));
            model.AddLayer(new Flatten(model.LastLayer));
            model.AddLayer(new Dense(model.LastLayer, 512, Activation.ELU));
            model.AddLayer(new Dense(model.LastLayer, 3, Activation.Softmax));
            net.Model = model;
            net.Optimize(new Adam(), Loss.Huber1);

            var input = new Tensor(new Shape(64, 64, 4, 32)); input.FillWithRand();
            var output = new Tensor(new Shape(1, 3, 1, 32));
            for (int n = 0; n < output.BatchSize; ++n)
                output[0, Tools.Rng.Next(output.Height), 0, n] = 1.0f;

            var timer = new Stopwatch();
            timer.Start();

            net.FitBatched(input, output, 10, 1, Track.Nothing);

            timer.Stop();
            Trace.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");
        }
    }
}
