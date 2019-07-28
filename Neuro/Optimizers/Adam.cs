using System;
using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Optimizers
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    public class Adam : OptimizerBase
    {
        public Adam(float lr = 0.001f)
        {
            LearningRate = lr;
        }

        protected override void OnStep(List<ParametersAndGradients> paramsAndGrads, int batchSize)
        {
            if (MGradients.Count != paramsAndGrads.Count)
            {
                for (var i = 0; i < paramsAndGrads.Count; i++)
                {
                    var gradients = paramsAndGrads[i].Gradients;

                    MGradients.Add(new Tensor(gradients.Shape));
                    VGradients.Add(new Tensor(gradients.Shape));
                }
            }

            for (var i = 0; i < paramsAndGrads.Count; i++)
            {
                var parametersAndGradient = paramsAndGrads[i];
                var parameters = parametersAndGradient.Parameters;
                var gradients = parametersAndGradient.Gradients;
                var mGrad = MGradients[i];
                var vGrad = VGradients[i];

                gradients.Div(batchSize, gradients);

                var tempLearningRate = LearningRate * (float)Math.Sqrt(1.0 - Math.Pow(Beta2, Iteration)) / (1.0f - (float)Math.Pow(Beta1, Iteration));
                
                //mGrad.Map((m, g) => m * Beta1 + (1 - Beta1) * g, gradients, mGrad);
                mGrad.Add(Beta1, 1 - Beta1, gradients, mGrad);
                vGrad.Map((v, g) => v * Beta2 + (1 - Beta2) * g * g, gradients, vGrad);

                parameters.Sub(mGrad.Div(vGrad.Map(x => (float)Math.Sqrt(x) + Epsilon)).Mul(tempLearningRate), parameters);

                gradients.Zero();
            }
        }

        public override string ToString()
        {
            return $"Adam(lr={LearningRate})";
            //return $"Adam(lr={LearningRate}, beta1={Beta1}, beta2={Beta2}, epsilon={Epsilon})";
        }

        private readonly float LearningRate;
        private readonly float Beta1 = 0.9f;
        private readonly float Beta2 = 0.999f;
        private readonly float Epsilon = 1e-8f;

        private List<Tensor> MGradients = new List<Tensor>();
        private List<Tensor> VGradients = new List<Tensor>();
    }
}
