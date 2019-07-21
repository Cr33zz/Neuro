using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;

namespace Neuro.Tests
{
    [TestClass]
    public class DenseLayerTests
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
            var layer = new Dense(10, 5, null);
            layer.Init();
            layer.Weights.FillWithRand();
            return layer;
        }
    }
}
