using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class GlorotUniform : InitializerBase
    {
        public GlorotUniform(double gain = 1)
        {
            Gain = gain;
        }

        public static double NextDouble(int fanIn, int fanOut, double gain)
        {
            double scale = 1 / Math.Max(1, (fanIn + fanOut) * 0.5);
            double limit = Math.Sqrt(3 * scale);
            return Uniform.NextDouble(-limit, limit);
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextDouble(fanIn, fanOut, Gain), t);
        }

        private readonly double Gain;
    }
}
