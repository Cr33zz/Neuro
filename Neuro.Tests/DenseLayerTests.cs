﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Layers;

namespace Neuro.Tests
{
    [TestClass]
    public class DenseLayerTests
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
            var layer = new Dense(10, 5, null);
            layer.Weights.FillWithRand();
            return layer;
        }
    }
}