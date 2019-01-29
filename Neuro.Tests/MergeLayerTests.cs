using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Tests
{
    [TestClass]
    public class MergeLayerTests
    {
        [TestMethod]
        public void Sum_InputGradient_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Sum));
        }

        [TestMethod]
        public void Sum_InputGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Sum), 3);
        }

        [TestMethod]
        public void Avg_InputGradient_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Avg));
        }

        [TestMethod]
        public void Avg_InputGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Avg), 3);
        }

        [TestMethod]
        public void Min_InputGradient_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Min));
        }

        [TestMethod]
        public void Min_InputGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Min), 3);
        }

        [TestMethod]
        public void Max_InputGradient_1Batch()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Max));
        }

        [TestMethod]
        public void Max_InputGradient_3Batches()
        {
            Tools.VerifyInputGradient(CreateLayer(Merge.Mode.Max), 3);
        }

        private LayerBase CreateLayer(Merge.Mode mode)
        {
            var inputShape = new Shape(1, 3);
            var layer = new Merge(new []{inputShape, inputShape}, mode);
            layer.ForceInit();
            return layer;
        }
    }
}
