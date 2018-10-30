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
    }

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    public class CategoricalCrossEntropy : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));
            targetOutput.Negated().MulElem(clippedOutput.Map(x => Math.Log(x)), result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));
            targetOutput.Negated().Div(clippedOutput, result);
        }
    }

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    public class CrossEntropy : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));
            targetOutput.Negated().MulElem(clippedOutput.Map(x => Math.Log(x))).Sub(targetOutput.Map(x => 1 - x).MulElem(clippedOutput.Map(x => Math.Log(1 - x))), result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));
            targetOutput.Negated().Div(clippedOutput).Add(targetOutput.Map(x => 1 - x).Div(clippedOutput.Map(x => 1 - x)), result);
        }
    }

    public class MeanSquareError : LossFunc
    {
        public override void Compute(Tensor targetOutput, Tensor output, Tensor result)
        {
            targetOutput.Sub(output, result);
            result.Map(x => x * x, result);
            result.Mul(0.5, result);
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            output.Sub(targetOutput, result);
        }
    }
}
