using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;
using System;
using System.Linq;

namespace Neuro.Tests
{
    public static class Tools
    {
        private static readonly double DERIVATIVE_EPSILON = 1e-4;
        private static readonly double LOSS_DERIVATIVE_EPSILON = 1e-5;

        public static void VerifyInputGradient(LayerBase layer, int batches = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batches));
            input.FillWithRand(7);
            var output = layer.FeedForward(input);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithValue(1);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            for (var i = 0; i < input.Shape.Length; ++i)
            {
                result.Zero();

                var oldValue = input.GetFlat(i);

                input.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                var output1 = layer.FeedForward(input).Clone();
                input.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                var output2 = layer.FeedForward(input).Clone();

                input.SetFlat(oldValue, i);

                output2.Sub(output1, result);

                var approxGrad = new double[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.GetFlat(j) / (2.0 * DERIVATIVE_EPSILON);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, layer.InputGradient.GetFlat(i), 1e-3, $"At element {i}");
            }
        }

        public static void VerifyParametersGradient(LayerBase layer, int batches = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batches));
            input.FillWithRand();
            var output = layer.FeedForward(input);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithValue(1);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            var parameters = layer.GetParameters();
            var parametersGradients = layer.GetParametersGradient();
            
            for (var i = 0; i < parameters.Shape.Length; i++)
            {
                result.Zero();

                double oldValue = parameters.GetFlat(i);
                parameters.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                var output1 = layer.FeedForward(input).Clone();
                parameters.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                var output2 = layer.FeedForward(input).Clone();

                parameters.SetFlat(oldValue, i);

                output1.Sub(output2, result);

                var approxGrad = new double[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.Get(j) / (2.0 * DERIVATIVE_EPSILON);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, parametersGradients.GetFlat(i), 1e-3, $"At element {i}");
            }
        }

        public static void VerifyActivationFuncDerivative(ActivationFunc func, int batches = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(3, 3, 3, batches));
            input.FillWithRange(-1.0, 2.0 / input.Length);

            var outputGradient = new Tensor(new Shape(3, 3, 3, batches));
            outputGradient.FillWithValue(1.0);

            // for derivation purposes activation functions expect already processed input
            var output = new Tensor(input.Shape);
            func.Compute(input, output);

            var derivative = new Tensor(input.Shape);
            func.Derivative(output, outputGradient, derivative);

            var output1 = new Tensor(input.Shape);
            func.Compute(input.Sub(DERIVATIVE_EPSILON), output1);

            var output2 = new Tensor(input.Shape);
            func.Compute(input.Add(DERIVATIVE_EPSILON), output2);

            var result = new Tensor(input.Shape);
            output2.Sub(output1, result);

            var approxDerivative = result.Div(2 * DERIVATIVE_EPSILON);

            Assert.IsTrue(approxDerivative.Equals(derivative, 1e-4));
        }

        public static void VerifyLossFuncDerivative(LossFunc func, Tensor targetOutput, int batches = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var output = new Tensor(new Shape(3, 3, 3, batches));
            output.FillWithRand(10);

            // for derivation purposes activation functions expect already processed input
            var error = new Tensor(output.Shape);
            func.Compute(targetOutput, output, error);

            var derivative = new Tensor(output.Shape);
            func.Derivative(targetOutput, output, derivative);

            var error1 = new Tensor(output.Shape);
            func.Compute(targetOutput, output.Sub(LOSS_DERIVATIVE_EPSILON), error1);

            var error2 = new Tensor(output.Shape);
            func.Compute(targetOutput, output.Add(LOSS_DERIVATIVE_EPSILON), error2);

            var result = new Tensor(output.Shape);
            error2.Sub(error1, result);

            var approxDerivative = result.Div(2 * LOSS_DERIVATIVE_EPSILON);

            Assert.IsTrue(approxDerivative.Equals(derivative, 1e-4));
        }

        public static void VerifyLossFunc(LossFunc func, Tensor targetOutput, Func<double, double, double> testFunc, int batches = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var output = new Tensor(new Shape(3, 3, 3, batches));
            output.FillWithRand(10);

            var error = new Tensor(output.Shape);
            func.Compute(targetOutput, output, error);

            for (int i = 0; i < output.Shape.Length; ++i)
                Assert.AreEqual(error.GetFlat(i), testFunc(targetOutput.GetFlat(i), output.GetFlat(i)), 1e-4);
        }
    }
}
