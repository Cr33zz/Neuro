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

            WeightsGradient = new Tensor(Weights.Shape);
            BiasGradient = new Tensor(Bias.Shape);
        }

        public override LayerBase Clone()
        {
            var clone = new Dense(InputShape.Length, OutputShape.Length, Activation);
            clone.Weights = Weights.Clone();
            clone.Bias = Bias.Clone();
            return clone;
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

        protected override void BackPropInternal(Tensor outputGradient)
        {
            // for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
            // each weight is responsible to the error in the next layer proportionally to its value
            Weights.Transposed().Mul(outputGradient, InputGradient);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Dense() errors gradient:\n{InputGradient}\n");

            var gradient = Optimizer != null ? Optimizer.GetGradientStep(outputGradient) : outputGradient;
            WeightsGradient.Add(gradient.Mul(Input.Transposed()).SumBatches(), WeightsGradient);
            BiasGradient.Add(gradient.SumBatches(), BiasGradient);
        }

        protected override void OnUpdateParameters(int trainingSamples)
        {
            WeightsGradient.Div(trainingSamples, WeightsGradient);
            Weights.Sub(WeightsGradient, Weights);
            BiasGradient.Div(trainingSamples, BiasGradient);
            Bias.Sub(BiasGradient, Bias);
        }

        protected override void ResetParametersGradients()
        {
            WeightsGradient.Zero();
            BiasGradient.Zero();
        }

        public override Tensor GetParameters() { return Weights; }

        public override Tensor GetParametersGradient() { return WeightsGradient; }

        public Tensor Weights;
        public Tensor Bias;

        public Initializers.InitializerBase KernelInitializer = new GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Zeros();

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
