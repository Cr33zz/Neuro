using System;
using System.Collections.Generic;
using Neuro.ComputationalGraph;
using Neuro.Tensors;

namespace Neuro.PerfTests
{
    class ComputationalGraphTest
    {
        static void Main(string[] args)
        {
            Graph graph = new Graph();
            var input1 = new Placeholder(new Shape(2));
            var input2 = new Placeholder(new Shape(2));

            var sum = Ops.add(input1, input2);

            var sess = new Session();
            Console.WriteLine(sess.Run(sum, new Dictionary<Placeholder, Tensor>{ {input1, new Tensor(new []{5.0f, 4.0f})}, { input2, new Tensor(new [] { 2.0f, 9.0f })} }));
        }
    }
}
