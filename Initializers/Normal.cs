using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class Normal : InitializerBase
    {
        public Normal(double mean = 0, double variance = 1, double scale = 1)
        {
            Mean = mean;
            Variance = variance;
            Scale = scale;
        }

        public static double NextDouble(double mean, double stdDeviation, double scale)
        {
            //based upon https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/randomkit.c
            double variance = stdDeviation * stdDeviation;

            if (HasValue)
            {
                HasValue = false;
                return (variance * (Value) + mean) * scale;
            }

            double x1, x2, r2;
            do
            {
                x1 = 2 * Tools.Rng.NextDouble() - 1;
                x2 = 2 * Tools.Rng.NextDouble() - 1;
                r2 = x1 * x1 + x2 * x2;
            }

            while (r2 >= 1.0 || r2 == 0.0);

            //Polar method, a more efficient version of the Box-Muller approach.
            double f = Math.Sqrt(-2 * Math.Log(r2) / r2);

            HasValue = true;
            Value = f * x1;

            return (variance * (f * x2) + mean) * scale;
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextDouble(Mean, Variance, Scale), t);
        }

        private readonly double Mean;
        private readonly double Variance;
        private readonly double Scale;

        private static bool HasValue = false;
        private static double Value;
    }
}
