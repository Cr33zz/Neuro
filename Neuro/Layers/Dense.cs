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
        public Dense(LayerBase inputLayer, int outputs, ActivationFunc activation = null)
            : base(inputLayer, new Shape(1, outputs), activation)
        {
        }

        // Use this constructor for input layer only!
        public Dense(int inputs, int outputs, ActivationFunc activation = null)
            : base(new Shape(1, inputs), new Shape(1, outputs), activation)
        {
        }

        // This constructor exists only for cloning purposes
        protected Dense()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Dense();
        }

        protected override void OnClone(LayerBase source)
        {
            base.OnClone(source);

            var sourceDense = source as Dense;
            Weights = sourceDense.Weights?.Clone();
            Bias = sourceDense.Bias?.Clone();
            UseBias = sourceDense.UseBias;
        }

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target, tau);

            var targetDense = target as Dense;
            Weights.CopyTo(targetDense.Weights, tau);
            Bias.CopyTo(targetDense.Bias, tau);
        }

        protected override void OnInit()
        {
			base.OnInit();

            Weights = new Tensor(new Shape(InputShape.Length, OutputShape.Length));
            Bias = new Tensor(OutputShape);

            WeightsGradient = new Tensor(Weights.Shape);
            BiasGradient = new Tensor(Bias.Shape);

            KernelInitializer.Init(Weights, InputShape.Length, OutputShape.Length);
            if (UseBias)
                BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return InputShape.Length * OutputShape.Length; }

        protected override void FeedForwardInternal()
        {
            Weights.Mul(Inputs[0], Output);
            if (UseBias)
                Output.Add(Bias, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            // for explanation watch https://www.youtube.com/watch?v=8H2ODPNxEgA&t=898s
            // each input is responsible for the output error proportionally to weights it is multiplied by
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
