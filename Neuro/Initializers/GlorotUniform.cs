using System;

namespace Neuro.Initializers
{
    public class GlorotUniform : InitializerBase
    {
        public GlorotUniform(float gain = 1)
        {
            Gain = gain;
        }

        public override Tensor Init(int[] shape)
        {
            (float fanIn, float fanOut) = ComputeFans(shape);
            float scale = 1 / (float)Math.Max(1, (fanIn + fanOut) * 0.5);
            float limit = (float)Math.Sqrt(3 * scale);
            return Backend.RandomUniform(shape, -limit, limit);    
        }

        private readonly float Gain;
    }
}
