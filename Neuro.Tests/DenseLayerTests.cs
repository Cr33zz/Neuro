using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class DenseLayerTests
    {
        [TestMethod]
        public void InputGradient_NoActivation()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var layer = new Dense(10, 5, null);
            layer.Weights.FillWithRand(100);
            Tools.VerifyInputGradient(layer);
        }

        [TestMethod]
        public void ParametersGradient_NoActivation()
        {
            Tensor.SetOpMode(Tensor.OpMode.CPU);
            var layer = new Dense(10, 5, null);
            layer.Weights.FillWithRand(100);
            Tools.VerifyParametersGradient(layer);
        }
    }
}
