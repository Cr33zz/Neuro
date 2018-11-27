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

        public Dense(LayerBase prevLayer, int outputs, ActivationFunc activation)
            : this(prevLayer.OutputShape.Length, outputs, activation)
        {
        }
        
        public Dense(int inputs, int outputs, ActivationFunc activation)
            : base(new Shape(1, inputs), new Shape(1, outputs), activation)
        {
            Weights = new Tensor(new Shape(inputs, outputs));
            Bias = new Tensor(OutputShape);

            WeightsGradient = new Tensor(Weights.Shape);
            BiasGradient = new Tensor(Bias.Shape);
        }

        public override LayerBase Clone()
        {
            var clone = new Dense(InputShape.Length, OutputShape.Length, Activation);
            clone.Weights = Weights.Clone();
            clone.Bias = Bias.Clone();
            clone.UseBias = UseBias;
            return clone;
        }

        public override void CopyParametersTo(LayerBase target)
        {
            base.CopyParametersTo(target);

            var targetDense = target as Dense;
            Weights.CopyTo(targetDense.Weights);
            Bias.CopyTo(targetDense.Bias);
        }

        public override void Init()
        {
            KernelInitializer.Init(Weights, InputShape.Length, OutputShape.Length);
            if (UseBias)
                BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return Weights.Length; }

        protected override void FeedForwardInternal()
        {
            Weights.Mul(Input, Output);
            if (UseBias)
                Output.Add(Bias, Output);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Dense() output:\n{Output}\n");
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            // for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
            // each weight is responsible to the error in the next layer proportionally to its value
            Weights.Transposed().Mul(outputGradient, InputGradient);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Dense() errors gradient:\n{InputGradient}\n");

            WeightsGradient.Add(outputGradient.Mul(Input.Transposed()).SumBatches(), WeightsGradient);
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
