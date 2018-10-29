using System;
using Neuro.Tensors;

namespace Neuro
{
    public static class Activation
    {
        //public static Tensor Linear(Tensor input, bool deriv = false)
        //{
        //    if (deriv)
        //        return Tensor.One(input.Depth, input.Height, input.Width);
            
        //    return new Tensor(input);
        //}

        public static void Sigmoid(Tensor input, bool deriv, Tensor result)
        {
            if (deriv)
            {
                input.Map(x => x * (1 - x), result); // we will call derivative for already sigmoieded values this is why we are not doing Sigmoid(x) * (1 - Sigmoid(x))
                return;
            }

            input.Map(x => 1 / (1 + Math.Exp(-x)), result);
        }

        public static void Tanh(Tensor input, bool deriv, Tensor result)
        {
            if (deriv)
            {
                input.Map(x => 1 - x * x, result);
                return;
            }

            input.Map(x => 2 / (1 + Math.Exp(-2 * x)) - 1, result);
        }

        public static void ReLU(Tensor input, bool deriv, Tensor result)
        {
            if (deriv)
            {
                input.Map(x => x > 0 ? 1 : 0, result);
                return;
            }

            input.Map(x => Math.Max(0, x), result);
        }

        public static void ELU(Tensor input, bool deriv, Tensor result)
        {
            const double ALPHA = 1;
            if (deriv)
            {
                input.Map(x => x > 0 ? 1 : (x + ALPHA), result);
                return;
            }

            input.Map(x => x >= 0 ? x : ALPHA * (Math.Exp(x) - 1), result);
        }

        public static void Softmax(Tensor input, bool deriv, Tensor result)
        {
            if (deriv)
            {
                input.Map(x => x * (1 - x), result);
                return;
            }

            Tensor shifted = input.Sub(input.Max());
            Tensor exps = shifted.Map(x => Math.Exp(x));

            for (int n = 0; n < input.Batches; ++n)
            {
                double sum = exps.Sum(n);

                for (int d = 0; d < input.Depth; ++d)
                for (int h = 0; h < input.Height; ++h)
                for (int w = 0; w < input.Width; ++w)
                    result[w, h, d, n] = exps[w, h, d, n] / sum;
            }
        }
    }
}
