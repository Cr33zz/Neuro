using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class GlorotNormal : InitializerBase
    {
        //It draws samples from a truncated normal distribution centered on 0 with deviation = sqrt(2 / (fanIn + fanOut))
        //where fanIn is the number of input units in the weight tensor and fanOut is the number of output units in the weight tensor.
        public GlorotNormal(float gain = 1)
        {
            Gain = gain;
        }

        public static float NextSingle(int fanIn, int fanOut, float gain)
        {
            float scale = 1 / (float)Math.Max(1, (fanIn + fanOut) * 0.5);
            float stdDev = gain * (float)Math.Sqrt(scale) / 0.87962566103423978f;
            return Normal.NextSingle(0, stdDev, 1);
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextSingle(fanIn, fanOut, Gain), t);
        }

        private readonly float Gain;
    }
}
