using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class FlattenLayerTests
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

        private LayerBase CreateLayer()
        {
            return new Flatten(new Shape(5, 5, 3));
        }
    }
}
