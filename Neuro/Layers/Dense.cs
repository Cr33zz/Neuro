using System;
using System.Diagnostics;
using System.Xml;
using Neuro.Initializers;
using Neuro.Tensors;

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

            WeightsDelta = new Tensor(Weights.Shape);
            BiasDelta = new Tensor(Bias.Shape);
        }

        public override void Init()
        {
            KernelInitializer.Init(Weights, InputShape.Length, OutputShape.Length);
            BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return Weights.Length; }

        protected override void FeedForwardInternal()
        {
            Weights.Mul(Input, Output);
            Output.Add(Bias, Output);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Dense() output:\n{Output}\n");
        }

        protected override void BackPropInternal(Tensor delta)
        {
            // for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
            // each weight is responsible to the error in the next layer proportionally to its value
            Weights.Transposed().Mul(delta, InputDelta);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Dense() errors gradient:\n{InputDelta}\n");

            var gradients = Optimizer.GetGradients(delta);
            WeightsDelta.Add(gradients.Mul(Input.Transposed()).SumBatches(), WeightsDelta);
            BiasDelta.Add(gradients.SumBatches(), BiasDelta);
        }

        protected override void OnUpdateParameters(int trainingSamples)
        {
            WeightsDelta.Div(trainingSamples, WeightsDelta);
            Weights.Sub(WeightsDelta, Weights);
            BiasDelta.Div(trainingSamples, BiasDelta);
            Bias.Sub(BiasDelta, Bias);
        }

        protected override void OnResetDeltas()
        {
            WeightsDelta.Zero();
            BiasDelta.Zero();
        }

        public Tensor Weights;
        public Tensor Bias;

        public Initializers.InitializerBase KernelInitializer = new GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Zeros();

        private Tensor WeightsDelta;
        private Tensor BiasDelta;

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
