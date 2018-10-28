using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class GlorotNormal : InitializerBase
    {
        //It draws samples from a truncated normal distribution centered on 0 with deviation = sqrt(2 / (fanIn + fanOut))
        //where fanIn is the number of input units in the weight tensor and fanOut is the number of output units in the weight tensor.
        public GlorotNormal(double gain = 1)
        {
            Gain = gain;
        }

        public static double NextDouble(int fanIn, int fanOut, double gain)
        {
            double scale = 1 / Math.Max(1, (fanIn + fanOut) * 0.5);
            double stdDev = gain * Math.Sqrt(scale) / 0.87962566103423978;
            return Normal.NextDouble(0, stdDev, 1);
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextDouble(fanIn, fanOut, Gain), t);
        }

        private readonly double Gain;
    }
}
