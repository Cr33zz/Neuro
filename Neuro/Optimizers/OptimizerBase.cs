using System.Collections.Generic;
using Tensorflow;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        protected OptimizerBase()
        {
            Iteration = tf.Variable(new Tensor(0), name: "optimizer_iteration");
        }

        public abstract List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss);

        public Tensor Iteration { get; protected set; }
    }
}
