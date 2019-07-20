using System;
using Tensorflow;

namespace Neuro
{
    public class GlorotNormal : InitializerBase
    {
        //It draws samples from a truncated normal distribution centered on 0 with deviation = sqrt(2 / (fanIn + fanOut))
        //where fanIn is the number of input units in the weight tensor and fanOut is the number of output units in the weight tensor.
        public GlorotNormal(float gain = 1)
        {
            Gain = gain;
        }

        public override Tensor Init(int[] shape, string name)
        {
            using (tf.name_scope(name + "glorot_normal"))
            {
                (float fanIn, float fanOut) = ComputeFans(shape);
                float scale = 1 / (float) Math.Max(1, (fanIn + fanOut) * 0.5);
                float stdDev = Gain * (float) Math.Sqrt(scale) / 0.87962566103423978f;
                return tf.random_normal(shape, 0, stdDev);
            }
        }

        private readonly float Gain;
    }
}
