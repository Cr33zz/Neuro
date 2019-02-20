using System.Collections.Generic;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        protected OptimizerBase()
        {
            Iteration = Backend.Variable(new Tensor(0), "optimizer_iteration");
        }

        public abstract List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss);

        public Tensor Iteration { get; protected set; }
    }
}
