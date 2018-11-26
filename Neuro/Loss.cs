using System;
using Neuro.Tensors;

namespace Neuro
{
    public abstract class LossFunc
    {
        public abstract void Compute(Tensor targetOutput, Tensor output, Tensor result);
        public abstract void Derivative(Tensor targetOutput, Tensor output, Tensor result);
    }

    public static class Loss
    {
        public static LossFunc CategoricalCrossEntropy = new CategoricalCrossEntropy();
        public static LossFunc CrossEntropy = new CrossEntropy();
        public static LossFunc MeanSquareError = new MeanSquareError();
        public static Huber Huber1 = new Huber(1);
    }

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    public class CategoricalCrossEntropy : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Clipped(Tools._EPSILON, 1 - Tools._EPSILON);
            targetOutput.Map((yTrue, y) => -yTrue * (float)Math.Log(y), clippedOutput, result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Clipped(Tools._EPSILON, 1 - Tools._EPSILON);
            targetOutput.Map((yTrue, y) => -yTrue / y, clippedOutput, result);
        }
    }

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    public class CrossEntropy : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Clipped(Tools._EPSILON, 1 - Tools._EPSILON);
            targetOutput.Map((yTrue, y) => -yTrue * (float)Math.Log(y) - (1 - yTrue) * (float)Math.Log(1 - y), clippedOutput, result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Clipped(Tools._EPSILON, 1 - Tools._EPSILON);
            targetOutput.Map((yTrue, y) => -yTrue / y + (1 - yTrue) / (1 - y), clippedOutput, result);
        }
    }

    public class MeanSquareError : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            targetOutput.Map((yTrue, y) => (y - yTrue) * (y - yTrue), output, result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            targetOutput.Map((yTrue, y) => (y - yTrue), output, result);
        }
    }

    public class Huber : LossFunc
    {
        public Huber(float delta)
        {
            Delta = delta;
        }

        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? (0.5f * a * a) : (Delta * (float)Math.Abs(a) - 0.5f * Delta * Delta); }, output, result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            targetOutput.Map((yTrue, y) => { float a = y - yTrue; return Math.Abs(a) <= Delta ? a : (Delta * Tools.Sign(a)); }, output, result);
        }

        public readonly float Delta;
    }
}
