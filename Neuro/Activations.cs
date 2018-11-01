using System;
using Neuro.Tensors;

namespace Neuro
{
    public abstract class ActivationFunc
    {
        public abstract void Compute(Tensor input, Tensor result);
        public abstract void Derivative(Tensor output, Tensor outputGradient, Tensor result);
    }

    public static class Activation
    {
        public static ActivationFunc Sigmoid = new Sigmoid();
        public static ActivationFunc Tanh = new Tanh();
        public static ActivationFunc ReLU = new ReLU();
        public static ActivationFunc ELU = new ELU();
        public static ActivationFunc Softmax = new Softmax();
    }

    public class Sigmoid : ActivationFunc
    {
        public override void Compute(Tensor input, Tensor result)
        {
            input.Map(x => 1 / (1 + Math.Exp(-x)), result);
        }

        public override void Derivative(Tensor output, Tensor outputGradient, Tensor result)
        {
            output.Map((x, x2) => x * (1 - x) * x2, outputGradient, result);
        }
    }

    public class Tanh : ActivationFunc
    {
        public override void Compute(Tensor input, Tensor result)
        {
            input.Map(x => 2 / (1 + Math.Exp(-2 * x)) - 1, result);
        }

        public override void Derivative(Tensor output, Tensor outputGradient, Tensor result)
        {
            output.Map((x, x2) => (1 - x * x) * x2, outputGradient, result);
        }
    }

    public class ReLU : ActivationFunc
    {
        public override void Compute(Tensor input, Tensor result)
        {
            input.Map(x => Math.Max(0, x), result);
        }

        public override void Derivative(Tensor output, Tensor outputGradient, Tensor result)
        {
            output.Map((x, x2) => x > 0 ? x2 : 0, outputGradient, result);
        }
    }

    public class ELU : ActivationFunc
    {
        private readonly double ALPHA = 1;

        public override void Compute(Tensor input, Tensor result)
        {
            input.Map(x => x >= 0 ? x : ALPHA * (Math.Exp(x) - 1), result);
        }

        public override void Derivative(Tensor output, Tensor outputGradient, Tensor result)
        {
            output.Map((x, x2) => (x > 0 ? 1 : (x + ALPHA)) * x2, outputGradient, result);
        }
    }

    public class Softmax : ActivationFunc
    {
        public override void Compute(Tensor input, Tensor result)
        {
            Tensor shifted = input.Sub(input.Max());
            Tensor exps = shifted.Map(x => Math.Exp(x));

            for (int n = 0; n < input.BatchSize; ++n)
            {
                double sum = exps.Sum(n);

                for (int d = 0; d < input.Depth; ++d)
                for (int h = 0; h < input.Height; ++h)
                for (int w = 0; w < input.Width; ++w)
                    result[w, h, d, n] = exps[w, h, d, n] / sum;
            }
        }

        public override void Derivative(Tensor output, Tensor outputGradient, Tensor result)
        {
            var outputReshaped = output.Reshaped(new Shape(1, Shape.Auto, 1, output.BatchSize));
            Tensor jacob = outputReshaped.DiagFlat().Sub(outputReshaped.Mul(outputReshaped.Transposed()));
            jacob.Mul(outputGradient, result);
        }
    }
}
