using System;
using System.Diagnostics;
using System.Xml;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Convolution : LayerBase
    {
        public Convolution(LayerBase prevLayer, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : this(prevLayer.OutputShape, filterSize, filtersNum, stride, activation)
        {
        }

        public Convolution(Shape inputShape, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : base(inputShape,
                   new Shape((int)Math.Floor((double)(inputShape.Width - filterSize) / stride + 1), (int)Math.Floor((double)(inputShape.Height - filterSize) / stride + 1), filtersNum),
                   activation)
        {
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;

            Weights = new Tensor(new Shape(FilterSize, FilterSize, inputShape.Depth, filtersNum));
            Bias = new Tensor(new Shape(OutputShape.Width, OutputShape.Height, filtersNum));
            WeightsDelta = new Tensor(Weights.Shape);
            BiasDelta = new Tensor(Bias.Shape);
        }

        public override void Init()
        {
            KernelInitializer.Init(Weights, InputShape.Length, OutputShape.Length);
            BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return FilterSize * FilterSize * FiltersNum; }

        protected override void FeedForwardInternal()
        {
            Input.Conv2D(Weights, Stride, Tensor.PaddingType.Valid, Output);
            Output.Add(Bias, Output);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Conv(f={FilterSize},s={Stride},filters={Weights.Length}) output:\n{Output}\n");
        }

        protected override void BackPropInternal(Tensor delta)
        {
            var gradients = Optimizer != null ? Optimizer.GetGradients(delta) : delta;

            Tensor.Conv2DInputsGradient(gradients, Weights, Stride, InputGradient);
            Tensor.Conv2DKernelsGradient(Output, Input, gradients, Stride, Tensor.PaddingType.Full, WeightsDelta);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Conv(f={FilterSize},s={Stride},filters={Weights.Length}) errors gradient:\n{InputGradient}\n");

            BiasDelta.Add(gradients.SumBatches());
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

        public override Tensor GetParameters() { return Weights; }

        public override Tensor GetParametersGradient() { return WeightsDelta; }

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

        public Tensor Weights;
        public Tensor Bias;

        public Tensor WeightsDelta;
        public Tensor BiasDelta;

        public Initializers.InitializerBase KernelInitializer = new Initializers.GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Initializers.Zeros();

        public int FiltersNum;
        public int FilterSize;
        public int Stride;
    }
}

