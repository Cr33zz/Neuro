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

            Kernels = new Tensor(new Shape(FilterSize, FilterSize, inputShape.Depth, filtersNum));
            Bias = new Tensor(new Shape(OutputShape.Width, OutputShape.Height, filtersNum));
            KernelsGradient = new Tensor(Kernels.Shape);
            BiasGradient = new Tensor(Bias.Shape);
        }

        public override LayerBase Clone()
        {
            var clone = new Convolution(InputShape, FilterSize, FiltersNum, Stride, Activation);
            clone.Kernels = Kernels.Clone();
            clone.Bias = Bias.Clone();
            return clone;
        }

        public override void Init()
        {
            KernelInitializer.Init(Kernels, InputShape.Length, OutputShape.Length);
            BiasInitializer.Init(Bias, InputShape.Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return FilterSize * FilterSize * FiltersNum; }

        protected override void FeedForwardInternal()
        {
            Input.Conv2D(Kernels, Stride, Tensor.PaddingType.Valid, Output);
            Output.Add(Bias, Output);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Conv(f={FilterSize},s={Stride},filters={Kernels.Length}) output:\n{Output}\n");
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            var gradient = Optimizer != null ? Optimizer.GetGradientStep(outputGradient) : outputGradient;

            Tensor.Conv2DInputsGradient(gradient, Kernels, Stride, InputGradient);
            Tensor.Conv2DKernelsGradient(Input, gradient, Stride, Tensor.PaddingType.Valid, KernelsGradient);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Conv(f={FilterSize},s={Stride},filters={Kernels.Length}) input gradient:\n{InputGradient}\n");

            BiasGradient.Add(gradient.SumBatches());
        }

        protected override void OnUpdateParameters(int trainingSamples)
        {
            KernelsGradient.Div(trainingSamples, KernelsGradient);
            Kernels.Sub(KernelsGradient, Kernels);
            BiasGradient.Div(trainingSamples, BiasGradient);
            Bias.Sub(BiasGradient, Bias);
        }

        protected override void ResetParametersGradients()
        {
            KernelsGradient.Zero();
            BiasGradient.Zero();
        }

        public override Tensor GetParameters() { return Kernels; }

        public override Tensor GetParametersGradient() { return KernelsGradient; }

        internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            Kernels.Serialize(elem, "Kernels");
            Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            Kernels.Deserialize(elem["Kernels"]);
            Bias.Deserialize(elem["Bias"]);
        }

        public Tensor Kernels;
        public Tensor Bias;

        public Tensor KernelsGradient;
        public Tensor BiasGradient;

        public Initializers.InitializerBase KernelInitializer = new Initializers.GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Initializers.Zeros();

        public int FiltersNum;
        public int FilterSize;
        public int Stride;
    }
}

