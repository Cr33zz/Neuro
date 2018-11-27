using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Optimizers;
using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Tests
{
    [TestClass]
    public class OptimizerTests
    {
        [TestMethod]
        public void SGD_Optimize()
        {
            TestOptimizer(new SGD());
        }

        [TestMethod]
        public void Adam_Optimize()
        {
            TestOptimizer(new Adam());
        }

        public void TestOptimizer(OptimizerBase optimizer)
        {
            Tensor input = new Tensor(new Shape(2, 2, 2, 2));
            input.FillWithRand(10);

            for (int i = 0; i < 10000; ++i)
            {
                optimizer.Step(new List<ParametersAndGradients>() { new ParametersAndGradients() { Parameters = input, Gradients = SquareFuncGradient(input) } }, 1);
            }

            var minimum = SquareFunc(input);

            for (int i = 0; i < input.Shape.Length; ++i)
                Assert.AreEqual(minimum.GetFlat(i), 0, 1e-5);
        }

        private Tensor SquareFuncGradient(Tensor input)
        {
            return input.Map(x => 2 * x);
        }

        private Tensor SquareFunc(Tensor input)
        {
            return input.Map(x => x * x);
        }
    }
}
