using System;
using Neuro.Tensors;

namespace Neuro
{
    public static class Loss
    {
        // This function can be used for any output being probability distribution (i.e. softmaxed)
        // https://gombru.github.io/2018/05/23/cross_entropy_loss/
        // The problem I have with that function is that when used along with softmax its derivative is not y_true - y_pred 
        // instead it will be zero for y_true being 0...
        public static void CategoricalCrossEntropy(Tensor targetOutput, Tensor output, bool deriv, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));
            if (deriv)
            {
                targetOutput.Negated().Div(clippedOutput, result);
                return;
            }

            targetOutput.Negated().MulElem(clippedOutput.Map(x => Math.Log(x)), result);
        }

        // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
        public static void CrossEntropy(Tensor targetOutput, Tensor output, bool deriv, Tensor result)
        {
            Tensor clippedOutput = output.Map(x => Tools.Clip(x, Tools._EPSILON, 1 - Tools._EPSILON));

            if (deriv)
            {
                targetOutput.Negated().Div(clippedOutput).Add(targetOutput.Map(x => 1 - x).Div(clippedOutput.Map(x => 1 - x)), result);
                return;
            }

            targetOutput.Negated().MulElem(clippedOutput.Map(x => Math.Log(x))).Sub(targetOutput.Map(x => 1 - x).MulElem(clippedOutput.Map(x => Math.Log(1 - x))), result);
        }

        //public static void SigmoidCrossEntropy(Tensor targetOutput, Tensor output, bool deriv, Tensor result)
        //{
        //    var sigmoidOutput = new Tensor(output.Shape);
        //    Sigmoid(output, false, sigmoidOutput);

        //    if (deriv)
        //    {
        //        sigmoidOutput.Sub(targetOutput, result);
        //        return;
        //    }

        //    targetOutput.MulElem(sigmoidOutput.Map(x => -Math.Log(x))).Sub(targetOutput.Map(x => 1 - x).MulElem(sigmoidOutput.Map(x => Math.Log(1 - x))), result);
        //}

        public static void MeanSquareError(Tensor targetOutput, Tensor output, bool deriv, Tensor result)
        {
            if (deriv)
            {
                output.Sub(targetOutput, result);
                return;
            }

            targetOutput.Sub(output, result);
            result.Map(x => x * x, result);
            result.Mul(0.5, result);
        }
    }
}
