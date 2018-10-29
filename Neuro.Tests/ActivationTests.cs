using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class ActivationTests
    {
        [TestMethod]
        public void Sigmoid_Derivative_1Batch()
        {
            Tools.VerifyFuncDerivative(Activation.Sigmoid);
        }

        [TestMethod]
        public void Sigmoid_Derivative_3Batches()
        {
            Tools.VerifyFuncDerivative(Activation.Sigmoid, 3);
        }

        [TestMethod]
        public void ReLU_Derivative_1Batch()
        {
            Tools.VerifyFuncDerivative(Activation.ReLU);
        }

        [TestMethod]
        public void ReLU_Derivative_3Batches()
        {
            Tools.VerifyFuncDerivative(Activation.ReLU, 3);
        }

        [TestMethod]
        public void Tanh_Derivative_1Batch()
        {
            Tools.VerifyFuncDerivative(Activation.Tanh);
        }

        [TestMethod]
        public void Tanh_Derivative_3Batches()
        {
            Tools.VerifyFuncDerivative(Activation.Tanh, 3);
        }

        [TestMethod]
        public void ELU_Derivative_1Batch()
        {
            Tools.VerifyFuncDerivative(Activation.ELU);
        }

        [TestMethod]
        public void ELU_Derivative_3Batches()
        {
            Tools.VerifyFuncDerivative(Activation.ELU, 3);
        }

        [TestMethod]
        public void Softmax_Derivative_1Batch()
        {
            Tools.VerifyFuncDerivative(Activation.Softmax);
        }

        [TestMethod]
        public void Softmax_Derivative_3Batches()
        {
            Tools.VerifyFuncDerivative(Activation.Softmax, 3);
        }

        [TestMethod]
        public void Softmax_1Batch()
        {
            var input = new Tensor(new Shape(3, 3, 3, 1));
            input.FillWithRand();

            var result = new Tensor(input.Shape);
            Activation.Softmax(input, false, result);

            Assert.AreEqual(result.Sum(0), 1, 1e-4);
        }

        [TestMethod]
        public void Softmax_3Batches()
        {
            var input = new Tensor(new Shape(3, 3, 3, 3));
            input.FillWithRand();

            var result = new Tensor(input.Shape);
            Activation.Softmax(input, false, result);

            for (int b = 0; b < 3; ++b)
                Assert.AreEqual(result.Sum(b), 1, 1e-4);
        }
    }
}
