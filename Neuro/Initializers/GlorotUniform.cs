using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class GlorotUniform : InitializerBase
    {
        public GlorotUniform(float gain = 1)
        {
            Gain = gain;
        }

        public static float NextSingle(int fanIn, int fanOut, float gain)
        {
            float scale = 1 / (float)Math.Max(1, (fanIn + fanOut) * 0.5);
            float limit = (float)Math.Sqrt(3 * scale);
            return Uniform.NextSingle(-limit, limit);
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => NextSingle(fanIn, fanOut, Gain), t);
        }

        private readonly float Gain;
    }
}
