using System;
using Neuro.Tensors;

namespace Neuro.Optimizers
{
    // Implementation based on https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
    public class Adam : OptimizerBase
    {
        public Adam(double lr = 0.001)
        {
            LearningRate = lr;
        }

        public override Tensor GetGradients(Tensor inputGradients)
        {
            if (M == null)
            {
                M = new Tensor(inputGradients.Shape);
                V = new Tensor(inputGradients.Shape);
            }

            ++T;
            M = M.Mul(Beta1).Add(inputGradients.Mul(1 - Beta1));
            V = V.Mul(Beta2).Add(inputGradients.Map(x => x * x).Mul(1 - Beta2));
            Tensor mCap = M.Div(1 - Math.Pow(Beta1, T));
            Tensor vCap = V.Div(1 - Math.Pow(Beta2, T));
            return mCap.Div(vCap.Map(x => Math.Sqrt(x)).Add(Epsilon)).Mul(LearningRate);
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
        private readonly double Epsilon = 1e-9;

        private Tensor M;
        private Tensor V;
        private double T;
    }
}
