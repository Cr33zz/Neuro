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
            Tools.VerifyInputGradient(CreateLayer());
        }

        [TestMethod]
        public void InputGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(), 3);
        }

        [TestMethod]
        public void ParametersGradient_1Batch()
        {
            Tools.VerifyParametersGradient(CreateLayer());
        }

        [TestMethod]
        public void ParametersGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(), 3);
        }

        private LayerBase CreateLayer()
        {
            var layer = new Convolution(new Shape(5,5,3), 3, 10, 1, null);
            layer.ForceInit();
            layer.Kernels.FillWithRand();
            return layer;
        }
    }
}
