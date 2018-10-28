using System;
using Neuro.Tensors;

namespace Neuro.Initializers
{
    public abstract class InitializerBase
    {
        public abstract void Init(Tensor t, int fanIn, int fanOut);
    }
}
