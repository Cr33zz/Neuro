using System.Diagnostics;
using System.Xml;
using Neuro.Initializers;
using Neuro.Tensors;
using System;
using System.Collections.Generic;

namespace Neuro.Layers
{
    public class Dense : LayerBase
    {
        // For serialization purposes only
        internal Dense() {}

        public Dense(LayerBase inputLayer, int outputs, ActivationFunc activation)
            : base(inputLayer, new Shape(1, outputs), activation)
        {
        }

        // Use this constructor for input layer only!
        public Dense(int inputs, int outputs, ActivationFunc activation)
            : base(new Shape(1, inputs), new Shape(1, outputs), activation)
        {
        }

        public override LayerBase Clone()
        {
            var clone = new Dense(InputShapes[0].Length, OutputShape.Length, Activation);
            clone.Weights = Weights.Clone();
            clone.Bias = Bias.Clone();
            clone.UseBias = UseBias;
            return clone;
        }

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target, tau);

            var targetDense = target as Dense;
            Weights.CopyTo(targetDense.Weights, tau);
            Bias.CopyTo(targetDense.Bias, tau);
        }

        protected override void Init()
        {
            Weights = new Tensor(new Shape(InputShape.Length, OutputShape.Length));
            Bias = new Tensor(OutputShape);

            WeightsGradient = new Tensor(Weights.Shape);
            BiasGradient = new Tensor(Bias.Shape);

            KernelInitializer.Init(Weights, InputShape.Length, OutputShape.Length);
            if (UseBias)
                BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return Weights.Length; }

        protected override void FeedForwardInternal()
        {
            Weights.Mul(Inputs[0], Output);
            if (UseBias)
                Output.Add(Bias, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            // for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
            // each weight is responsible to the error in the next layer proportionally to its value
            Weights.Transposed().Mul(outputGradient, InputsGradient[0]);

            WeightsGradient.Add(outputGradient.Mul(Inputs[0].Transposed()).SumBatches(), WeightsGradient);
            if (UseBias)
                BiasGradient.Add(outputGradient.SumBatches(), BiasGradient);
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var result = new List<ParametersAndGradients>();

            result.Add(new ParametersAndGradients() { Parameters = Weights, Gradients = WeightsGradient });

            if (UseBias)
                result.Add(new ParametersAndGradients() { Parameters = Bias, Gradients = BiasGradient });

            return result;
        }

        public Tensor Weights;
        public Tensor Bias;
        public bool UseBias = true;

        public InitializerBase KernelInitializer = new GlorotUniform();
        public InitializerBase BiasInitializer = new Zeros();

        private Tensor WeightsGradient;
        private Tensor BiasGradient;

        internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            Weights.Serialize(elem, "Weights");
            Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            Weights.Deserialize(elem["Weights"]);
            Bias.Deserialize(elem["Bias"]);
        }
    }
}
