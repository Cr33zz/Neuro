using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Optimizers
{
    public class SGD : OptimizerBase
    {
        public SGD(float lr = 0.01f)
        {
            LearningRate = lr;
        }

        protected override void OnStep(List<ParametersAndGradients> paramsAndGrads, int batchSize)
        {
            for (var i = 0; i < paramsAndGrads.Count; i++)
            {
                var parametersAndGradient = paramsAndGrads[i];
                var parameters = parametersAndGradient.Parameters;
                var gradients = parametersAndGradient.Gradients;

                var tempLearningRate = LearningRate/* / batchSize*/;

                gradients.Mul(tempLearningRate, gradients);
                parameters.Sub(gradients, parameters);

                gradients.Zero();
            }
        }

        public override string ToString()
        {
            return $"SGD(lr={LearningRate})";
        }

        private readonly float LearningRate;
    }
}
