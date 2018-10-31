using Neuro.Tensors;

namespace Neuro.Optimizers
{
    public class SGD : OptimizerBase
    {
        public SGD(double lr = 0.02)
        {
            LearningRate = lr;
        }

        public override Tensor GetGradientStep(Tensor gradient)
        {
            return gradient.Mul(LearningRate);
        }

        public override OptimizerBase Clone()
        {
            return new SGD(LearningRate);
        }

        public override string ToString()
        {
            return $"SGD(lr={LearningRate})";
        }

        private readonly double LearningRate;
    }
}
