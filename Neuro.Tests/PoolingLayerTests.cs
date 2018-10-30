using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class PoolingLayerTests
    {
        [TestMethod]
        public void InputGradient_MaxPooling_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Tensor.PoolType.Max));
        }

        [TestMethod]
        public void InputGradient_MaxPooling_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Tensor.PoolType.Max), 3);
        }

        [TestMethod]
        public void InputGradient_AvgPooling_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Tensor.PoolType.Avg));
        }

        [TestMethod]
        public void InputGradient_AvgPooling_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Tensor.PoolType.Avg), 3);
        }

        private LayerBase CreateLayer(Tensor.PoolType poolType)
        {
            return new Pooling(new Shape(6, 6, 3), 2, 2, poolType);
        }
    }
}
