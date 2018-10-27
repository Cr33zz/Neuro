using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class Uniform : InitializerBase
    {
        public Uniform(double min = -0.05, double max = 0.05)
        {
            Min = min;
            Max = max;
        }

        public static double NextDouble(double min, double max)
        {
            return min + Tools.Rng.NextDouble() * (max - min);
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextDouble(Min, Max), t);
        }

        private readonly double Min;
        private readonly double Max;
    }
}
