using System;
using Neuro.Tensors;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        public abstract Tensor GetGradientStep(Tensor gradient);
        public abstract OptimizerBase Clone();        
    }
}
