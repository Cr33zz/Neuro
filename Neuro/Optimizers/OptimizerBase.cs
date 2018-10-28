using System;
using Neuro.Tensors;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        public abstract Tensor GetGradients(Tensor inputGradients);
        public abstract OptimizerBase Clone();        
    }
}
