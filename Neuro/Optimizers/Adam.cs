using System;
using Neuro.Tensors;

namespace Neuro.Optimizers
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    public class Adam : OptimizerBase
    {
        public Adam(float lr = 0.001f)
        {
            LearningRate = lr;
        }

        public override Tensor GetGradientStep(Tensor gradient)
        {
            if (M == null)
            {
                M = new Tensor(gradient.Shape);
                V = new Tensor(gradient.Shape);
            }

            var tempLearningRate = LearningRate * (float)Math.Sqrt(1.0 - Math.Pow(Beta2, Iteration)) / (1.0f - (float)Math.Pow(Beta1, Iteration));

            M.Map((m, g) => m * Beta1 + (1 - Beta1) * g, gradient, M);
            V.Map((v, g) => v * Beta2 + (1 - Beta2) * g * g, gradient, V);

            return M.Div(V.Map(x => (float)Math.Sqrt(x) + Epsilon)).Mul(tempLearningRate);
        }

        public override OptimizerBase Clone()
        {
            return new Adam(LearningRate);
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

        private Tensor M;
        private Tensor V;
    }
}
