using System;
using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Optimizers
{
    public abstract class OptimizerBase
    {
        public void Step(List<ParametersAndGradients> paramsAndGrads, int batchSize)
        {
            ++Iteration;

            OnStep(paramsAndGrads, batchSize);
        }

        protected abstract void OnStep(List<ParametersAndGradients> paramsAndGrads, int batchSize);

        public float Iteration { get; protected set; }
    }
}
