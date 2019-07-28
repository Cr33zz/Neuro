using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class ConvolutionLayerTests
    {
        [TestMethod]
        public void InputGradient_1Batch()
        {
            Assert.IsTrue(TestTools.VerifyInputGradient(CreateLayer()));
        }

        [TestMethod]
        public void InputGradient_3Batches()
        {
            Assert.IsTrue(TestTools.VerifyInputGradient(CreateLayer(), 3));
        }

        [TestMethod]
        public void ParametersGradient_1Batch()
        {
            Assert.IsTrue(TestTools.VerifyParametersGradient(CreateLayer()));
        }

        [TestMethod]
        public void ParametersGradient_3Batches()
        {
            Assert.IsTrue(TestTools.VerifyInputGradient(CreateLayer(), 3));
        }

        private LayerBase CreateLayer()
        {
            var layer = new Convolution(new Shape(5,5,3), 3, 10, 1, null);
            layer.Init();
            layer.Kernels.FillWithRand();
            return layer;
        }
    }
}
