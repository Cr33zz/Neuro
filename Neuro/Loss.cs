using System;
using TensorFlow;

namespace Neuro
{
    public abstract class Loss
    {
        public abstract Tensor Build(Tensor targetOutput, Tensor output);
    }

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    public class CategoricalCrossEntropy : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (Backend.WithScope("categorical_cross_entropy"))
            {
                var axis = output.Shape.ToIntArray().Get(-1);
                var clippedOutput = Backend.ClipByValue(output, Tools._EPSILON, 1 - Tools._EPSILON);
                return Backend.Neg(Backend.ReduceSum(Backend.Mul(targetOutput, Backend.Log(clippedOutput)), axis));
            }
        }
    }

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    public class CrossEntropy : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (Backend.WithScope("cross_entropy"))
            {
                var axis = output.Shape.ToIntArray().Get(-1);
                var clippedOutput = Backend.ClipByValue(output, Tools._EPSILON, 1 - Tools._EPSILON);
                return Backend.Sub(Backend.Neg(Backend.ReduceSum(Backend.Mul(targetOutput, Backend.Log(clippedOutput)), axis)), 
                                   Backend.Neg(Backend.ReduceSum(Backend.Mul(Backend.Sub(1, targetOutput), Backend.Log(Backend.Sub(1, clippedOutput))), axis)));
            }
        }
    }

    public class MeanSquareError : Loss
    {
        public override Tensor Build(Tensor targetOutput, Tensor output)
        {
            using (Backend.WithScope("mean_square_error"))
                return Backend.Mean(Backend.Square(output - targetOutput));
        }
    }

    /*public class Huber : LossFunc
    {
        public Huber(float delta)
        {
            Delta = delta;
        }

        public override void Compute(TFTensor targetOutput, TFTensor output, TFTensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? (0.5f * a * a) : (Delta * (float)Math.Abs(a) - 0.5f * Delta * Delta); }, output, result);
        }

        public override void Derivative(TFTensor targetOutput, TFTensor output, TFTensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? a : (Delta * Tools.Sign(a)); }, output, result);
        }

        public readonly float Delta;
    }*/
}
