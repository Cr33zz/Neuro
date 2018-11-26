using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class Normal : InitializerBase
    {
        public Normal(float mean = 0, float variance = 1, float scale = 1)
        {
            Mean = mean;
            Variance = variance;
            Scale = scale;
        }

        public static float NextSingle(float mean, float stdDeviation, float scale)
        {
            //based upon https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/randomkit.c
            float variance = stdDeviation * stdDeviation;

            if (HasValue)
            {
                HasValue = false;
                return (variance * (Value) + mean) * scale;
            }

            float x1, x2, r2;
            do
            {
                x1 = 2 * (float)Tools.Rng.NextDouble() - 1;
                x2 = 2 * (float)Tools.Rng.NextDouble() - 1;
                r2 = x1 * x1 + x2 * x2;
            }

            while (r2 >= 1.0 || r2 == 0.0);

            //Polar method, a more efficient version of the Box-Muller approach.
            float f = (float)Math.Sqrt(-2 * Math.Log(r2) / r2);

            HasValue = true;
            Value = f * x1;

            return (variance * (f * x2) + mean) * scale;
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextSingle(Mean, Variance, Scale), t);
        }

        private readonly float Mean;
        private readonly float Variance;
        private readonly float Scale;

        private static bool HasValue = false;
        private static float Value;
    }
}
