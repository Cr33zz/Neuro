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
            targetOutput.SetFlat(1.0f, 1);
            Assert.IsTrue(TestTools.VerifyLossFunc(Loss.CrossEntropy, targetOutput, (yTrue, y) => -yTrue * (float)Math.Log(y) - (1 - yTrue) * (float)Math.Log(1 - y), true, 3));
        }

        [TestMethod]
        public void CrossEntropy_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0f, 1);
            Assert.IsTrue(TestTools.VerifyLossFuncDerivative(Loss.CrossEntropy, targetOutput, true, 3, 0.1f));
        }

        [TestMethod]
        public void CategoricalCrossEntropy_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0f, 1);
            Assert.IsTrue(TestTools.VerifyLossFunc(Loss.CategoricalCrossEntropy, targetOutput, (yTrue, y) => -yTrue * (float)Math.Log(y), true, 3));
        }

        [TestMethod]
        public void CategoricalCrossEntropy_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.SetFlat(1.0f, 1);
            Assert.IsTrue(TestTools.VerifyLossFuncDerivative(Loss.CategoricalCrossEntropy, targetOutput, true, 3));
        }

        [TestMethod]
        public void MeanSquareError_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10);
            Assert.IsTrue(TestTools.VerifyLossFunc(Loss.MeanSquareError, targetOutput, (yTrue, y) => (yTrue - y) * (yTrue - y) * 0.5f, false, 3));
        }

        // I followed the same implementation as Keras
        //[TestMethod]
        //public void MeanSquareError_Derivative()
        //{
        //    var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
        //    targetOutput.FillWithRand(10, -2, 2);
        //    Assert.IsTrue(TestTools.VerifyLossFuncDerivative(Loss.MeanSquareError, targetOutput, false, 3));
        //}

        [TestMethod]
        public void Huber_Compute()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10);
            Assert.IsTrue(TestTools.VerifyLossFunc(Loss.Huber1, targetOutput, (yTrue, y) => { float a = yTrue - y; return Math.Abs(a) <= Loss.Huber1.Delta ? (0.5f * a * a) : (Loss.Huber1.Delta * (float)(Math.Abs(a) - 0.5 * Loss.Huber1.Delta)); }, false, 3));
        }

        [TestMethod]
        public void Huber_Derivative()
        {
            var targetOutput = new Tensor(new Shape(3, 3, 3, 3));
            targetOutput.FillWithRand(10, -2, 2);
            Assert.IsTrue(TestTools.VerifyLossFuncDerivative(Loss.Huber1, targetOutput, false, 3));
        }
    }
}
