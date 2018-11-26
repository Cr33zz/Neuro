using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class Constant : InitializerBase
    {
        public Constant(float value = 1)
        {
            Value = value;
        }

        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Map(x => Value, t);
        }

        private readonly float Value;
    }
}
