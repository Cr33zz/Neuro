using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;
using System.Linq;

namespace Neuro.Tests
{
    public class Tools
    {
        public static void VerifyInputGradient(LayerBase layer, int batchSize = 1, double epsilon = 1e-4)
        {
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batchSize));
            input.FillWithRand(7);
            var output = layer.FeedForward(input);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithRand(8);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            for (var b = 0; b < batchSize; ++b)
            for (var d = 0; d < layer.InputShape.Depth; ++d)
            for (var h = 0; h < layer.InputShape.Height; ++h)
            for (var w = 0; w < layer.InputShape.Width; ++w)
            {
                result.Zero();

                var oldValue = input[w, h, d, b];

                input[w, h, d, b] = oldValue + epsilon;
                var output1 = layer.FeedForward(input).Clone();
                input[w, h, d, b] = oldValue - epsilon;
                var output2 = layer.FeedForward(input).Clone();

                input[w, h, d, b] = oldValue;

                output1.Sub(output2, result);

                var approxGrad = new double[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.GetFlat(j) / (2.0 * epsilon);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, layer.InputGradient[w, h, d, b], 1e-3);
            }
        }

        public static void VerifyParametersGradient(LayerBase layer, int batchSize = 1, double epsilon = 1e-4)
        {
            var input = new Tensor(new Shape(layer.InputShape.Width, layer.InputShape.Height, layer.InputShape.Depth, batchSize));
            input.FillWithRand(7);
            var output = layer.FeedForward(input);
            var outputGradient = new Tensor(output.Shape);
            outputGradient.FillWithRand(8);

            layer.BackProp(outputGradient);

            var result = new Tensor(output.Shape);

            var parameters = layer.GetParameters();
            var parametersGradients = layer.GetParametersGradient();
            
            for (var i = 0; i < parameters.Shape.Length; i++)
            {
                result.Zero();

                double oldValue = parameters.GetFlat(i);
                parameters.SetFlat(oldValue + epsilon, i);
                var output1 = layer.FeedForward(input).Clone();
                parameters.SetFlat(oldValue - epsilon, i);
                var output2 = layer.FeedForward(input).Clone();

                parameters.SetFlat(oldValue, i);

                output1.Sub(output2, result);

                var approxGrad = new double[output.Shape.Length];
                for (var j = 0; j < output.Shape.Length; j++)
                    approxGrad[j] = result.Get(j) / (2.0 * epsilon);

                var approxGradient = approxGrad.Sum();
                Assert.AreEqual(approxGradient, parametersGradients.GetFlat(i), 1e-3);
            }
        }
    }
}
