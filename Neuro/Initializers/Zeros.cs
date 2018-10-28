using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public class Zeros : InitializerBase
    {
        public override void Init(Tensor t, int fanIn, int fanOut)
        {
            t.Zero();
        }
    }
}
