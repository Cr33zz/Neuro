﻿using System;
using System.Diagnostics;
using System.Linq;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro
{
    public static class TestTools
    {
        private static readonly float DERIVATIVE_EPSILON = 1e-4f;
        private static readonly float LOSS_DERIVATIVE_EPSILON = 1e-5f;

        public static bool ValidateLayer(LayerBase layer)
        {
            return VerifyInputGradient(layer) && VerifyInputGradient(layer, 3) &&
                   VerifyParametersGradient(layer) && VerifyParametersGradient(layer, 3);
        }

        public static bool VerifyInputGradient(LayerBase layer, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var inputs = GenerateInputsForLayer(layer, batchSize);

            var output = layer.FeedForward(inputs);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithValue(1);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            for (int n = 0; n < inputs.Length; ++n)
            {
                var input = inputs[n];
                for (int i = 0; i < input.Shape.Length; ++i)
                {
                    result.Zero();

                    var oldValue = input.GetFlat(i);

                    input.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                    var output1 = layer.FeedForward(inputs).Clone();
                    input.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                    var output2 = layer.FeedForward(inputs).Clone();

                    input.SetFlat(oldValue, i);

                    output2.Sub(output1, result);

                    var approxGrad = new float[output.Shape.Length];
                    float approxGradient = 0;
                    for (int j = 0; j < output.Shape.Length; j++)
                    {
                        approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);
                        approxGradient += approxGrad[j];
                    }

                    if (Math.Abs(approxGradient - layer.InputsGradient[n].GetFlat(i)) > 0.02)
                    {
                        Debug.Assert(false, $"Input gradient validation failed at element {i} of input {n}, expected {approxGradient} actual {layer.InputsGradient[n].GetFlat(i)}!");
                        return false;
                    }
                }
            }

            return true;
        }

        public static bool VerifyParametersGradient(LayerBase layer, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var inputs = GenerateInputsForLayer(layer, batchSize);

            var output = layer.FeedForward(inputs);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithValue(1);

            layer.BackProp(outputGradient);

            var paramsAndGrads = layer.GetParametersAndGradients();

            if (paramsAndGrads.Count == 0)
                return true;

            var result = new Tensor(output.Shape);

            var parameters = paramsAndGrads[0].Parameters;
            var gradients = paramsAndGrads[0].Gradients;

            for (var i = 0; i < parameters.Shape.Length; i++)
            {
                result.Zero();

                float oldValue = parameters.GetFlat(i);
                parameters.SetFlat(oldValue + DERIVATIVE_EPSILON, i);
                var output1 = layer.FeedForward(inputs).Clone();
                parameters.SetFlat(oldValue - DERIVATIVE_EPSILON, i);
                var output2 = layer.FeedForward(inputs).Clone();

                parameters.SetFlat(oldValue, i);

                output1.Sub(output2, result);

                var approxGrad = new float[output.Shape.Length];
                for (int j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.GetFlat(j) / (2.0f * DERIVATIVE_EPSILON);

                var approxGradient = approxGrad.Sum();
                if (Math.Abs(approxGradient - gradients.GetFlat(i)) > 0.02)
                {
                    Debug.Assert(false, $"Parameter gradient validation failed at parameter {i}, expected {approxGradient} actual {gradients.GetFlat(i)}!");
                    return false;
                }
            }

            return true;
        }

        private static Tensor[] GenerateInputsForLayer(LayerBase layer, int batchSize)
        {
            var inputs = new Tensor[layer.InputShapes.Length];

            for (int i = 0; i < inputs.Length; ++i)
            {
                var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batchSize));
                input.FillWithRand(7 + i);
                inputs[i] = input;
            }

            return inputs;
        }

        public static bool VerifyActivationFuncDerivative(ActivationFunc func, int batchSize = 1)
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

            return approxDerivative.Equals(derivative, 1e-2f);
        }

        public static bool VerifyLossFuncDerivative(LossFunc func, Tensor targetOutput, bool onlyPositiveOutput = false, int batchSize = 1, float tolerance = 0.01f)
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

            return approxDerivative.Equals(derivative, tolerance);
        }

        public static bool VerifyLossFunc(LossFunc func, Tensor targetOutput, Func<float, float, float> testFunc, bool onlyPositiveOutput = false, int batchSize = 1)
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var output = new Tensor(new Shape(3, 3, 3, batchSize));
            output.FillWithRand(10, onlyPositiveOutput ? 0 : -1);

            var error = new Tensor(output.Shape);
            func.Compute(targetOutput, output, error);

            for (int i = 0; i < output.Shape.Length; ++i)
            {
                if (Math.Abs(error.GetFlat(i) - testFunc(targetOutput.GetFlat(i), output.GetFlat(i))) > 1e-4)
                    return false;
            }

            return true;
        }
    }
}
