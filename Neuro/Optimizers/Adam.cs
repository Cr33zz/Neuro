using System;
using Neuro.Tensors;

namespace Neuro.Optimizers
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    public class Adam : OptimizerBase
    {
        public Adam(double lr = 0.001)
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

            ++T;

            var tempLearningRate = LearningRate * Math.Sqrt(1.0 - Math.Pow(Beta2, T)) / (1.0 - Math.Pow(Beta1, T));

            M.Mul(Beta1).Add(gradient.Map(g => (1 - Beta1) * g), M);
            V.Mul(Beta2).Add(gradient.Map(g => (1 - Beta2) * g * g), V);

            return M.Div(V.Map(x => Math.Sqrt(x) + Epsilon)).Mul(tempLearningRate);
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

        private readonly double LearningRate;
        private readonly double Beta1 = 0.9;
        private readonly double Beta2 = 0.999;
        private readonly double Epsilon = 1e-8;

        private Tensor M;
        private Tensor V;
        private double T;
    }
}
