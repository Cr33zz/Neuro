using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;
using System;
using System.Linq;

namespace Neuro.Tests
{
    public static class Tools
    {
        private static readonly float DERIVATIVE_EPSILON = 1e-4f;
        private static readonly float LOSS_DERIVATIVE_EPSILON = 1e-5f;

        public static void VerifyInputGradient(LayerBase layer, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batchSize));
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

                var approxGrad = new float[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, layer.InputGradient.GetFlat(i), 0.02, $"At element {i}");
            }
        }

        public static void VerifyParametersGradient(LayerBase layer, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batchSize));
            input.FillWithRand(7);
            var output = layer.FeedForward(input);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithValue(1);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            var paramsAndGrads = layer.GetParametersAndGradients();
            var weights = paramsAndGrads[0].Parameters;
            var weightsGrads = paramsAndGrads[0].Gradients;

            for (var i = 0; i < weights.Shape.Length; i++)
            {
                result.Zero();

                float oldValue = weights.GetFlat(i);
                weights.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                var output1 = layer.FeedForward(input).Clone();
                weights.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                var output2 = layer.FeedForward(input).Clone();

                weights.SetFlat(oldValue, i);

                output1.Sub(output2, result);

                var approxGrad = new float[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, weightsGrads.GetFlat(i), 0.02, $"At element {i}");
            }
        }

        public static void VerifyActivationFuncDerivative(ActivationFunc func, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var input = new Tensor(new Shape(3, 3, 3, batchSize));
            input.FillWithRange(-1.0f, 2.0f / input.Length);

            var outputGradient = new Tensor(new Shape(3, 3, 3, batchSize));
            outputGradient.FillWithValue(1.0f);

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

            Assert.IsTrue(approxDerivative.Equals(derivative, 1e-2f));
        }

        public static void VerifyLossFuncDerivative(LossFunc func, Tensor targetOutput, bool onlyPositiveOutput = false, int batchSize = 1, float tolerance = 0.01f)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var output = new Tensor(new Shape(3, 3, 3, batchSize));
            output.FillWithRand(10, onlyPositiveOutput ? 0 : -1);

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

            Assert.IsTrue(approxDerivative.Equals(derivative, tolerance));
        }

        public static void VerifyLossFunc(LossFunc func, Tensor targetOutput, Func<float, float, float> testFunc, bool onlyPositiveOutput = false, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var output = new Tensor(new Shape(3, 3, 3, batchSize));
            output.FillWithRand(10, onlyPositiveOutput ? 0 : -1);

            var error = new Tensor(output.Shape);
            func.Compute(targetOutput, output, error);

            for (int i = 0; i < output.Shape.Length; ++i)
                Assert.AreEqual(error.GetFlat(i), testFunc(targetOutput.GetFlat(i), output.GetFlat(i)), 1e-4);
        }
    }
}
