using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class LossTests
    {
        [TestMethod]
        public void CrossEntropy_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0, 1);
            Tools.VerifyLossFunc(Loss.CrossEntropy, targetOutput, (yTrue, y) => -yTrue * Math.Log(y) - (1 - yTrue) * Math.Log(1 - y), true, 3);
        }

        [TestMethod]
        public void CrossEntropy_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0, 1);
            Tools.VerifyLossFuncDerivative(Loss.CrossEntropy, targetOutput, true, 3);
        }

        [TestMethod]
        public void CategoricalCrossEntropy_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0, 1);
            Tools.VerifyLossFunc(Loss.CategoricalCrossEntropy, targetOutput, (yTrue, y) => -yTrue * Math.Log(y), true, 3);
        }

        [TestMethod]
        public void CategoricalCrossEntropy_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0, 1);
            Tools.VerifyLossFuncDerivative(Loss.CategoricalCrossEntropy, targetOutput, true, 3);
        }

        [TestMethod]
        public void MeanSquareError_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10);
            Tools.VerifyLossFunc(Loss.MeanSquareError, targetOutput, (yTrue, y) => (yTrue - y) * (yTrue - y) * 0.5, false, 3);
        }

        [TestMethod]
        public void MeanSquareError_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10, -2, 2);
            Tools.VerifyLossFuncDerivative(Loss.MeanSquareError, targetOutput, false, 3);
        }

        [TestMethod]
        public void Huber_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10);
            Tools.VerifyLossFunc(Loss.Huber1, targetOutput, (yTrue, y) => { double a = yTrue - y; return Math.Abs(a) <= Loss.Huber1.Delta ? (0.5 * a * a) : (Loss.Huber1.Delta * (Math.Abs(a) - 0.5 * Loss.Huber1.Delta)); }, false, 3);
        }

        [TestMethod]
        public void Huber_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10, -2, 2);
            Tools.VerifyLossFuncDerivative(Loss.Huber1, targetOutput, false, 3);
        }
    }
}
