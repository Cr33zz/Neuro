using System.Collections.Generic;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        protected abstract List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss);

        public Tensor Iteration { get; protected set; }
    }
}
