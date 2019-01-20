using System;
using System.Diagnostics;
using System.Xml;
using Neuro.Tensors;
using System.Collections.Generic;

namespace Neuro.Layers
{
    public class Convolution : LayerBase
    {
        public Convolution(LayerBase prevLayer, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : base(prevLayer, GetOutShape(prevLayer.OutputShape, filterSize, filterSize, stride, filtersNum), activation)
        {
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;
        }

        // Use this constructor for input layer only!
        public Convolution(Shape inputShape, int filterSize, int filtersNum, int stride, ActivationFunc activation)
            : base(inputShape, GetOutShape(inputShape, filterSize, filterSize, stride, filtersNum), activation)
        {
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;
        }

        public override LayerBase Clone()
        {
            var clone = new Convolution(InputShapes[0], FilterSize, FiltersNum, Stride, Activation);
            clone.Kernels = Kernels.Clone();
            clone.Bias = Bias.Clone();
            clone.UseBias = UseBias;
            return clone;
        }

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target);

            var targetConv = target as Convolution;
            Kernels.CopyTo(targetConv.Kernels, tau);
            Bias.CopyTo(targetConv.Bias, tau);
        }

        protected override void Init()
        {
            Kernels = new Tensor(new Shape(FilterSize, FilterSize, InputShape.Depth, FiltersNum));
            Bias = new Tensor(new Shape(OutputShape.Width, OutputShape.Height, FiltersNum));
            KernelsGradient = new Tensor(Kernels.Shape);
            BiasGradient = new Tensor(Bias.Shape);

            KernelInitializer.Init(Kernels, InputShapes[0].Length, OutputShape.Length);
            if (UseBias)
                BiasInitializer.Init(Bias, InputShapes[0].Length, OutputShape.Length);
        }

        public override int GetParamsNum() { return FilterSize * FilterSize * FiltersNum; }

        protected override void FeedForwardInternal()
        {
            Inputs[0].Conv2D(Kernels, Stride, Tensor.PaddingType.Valid, Output);
            if (UseBias)
                Output.Add(Bias, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            Tensor.Conv2DInputsGradient(outputGradient, Kernels, Stride, InputsGradient[0]);
            Tensor.Conv2DKernelsGradient(Inputs[0], outputGradient, Stride, Tensor.PaddingType.Valid, KernelsGradient);

            if (UseBias)
                BiasGradient.Add(outputGradient.SumBatches());
        }

        public override List<ParametersAndGradients> GetParametersAndGradients()
        {
            var result = new List<ParametersAndGradients>();

            result.Add(new ParametersAndGradients() { Parameters = Kernels, Gradients = KernelsGradient });

            if (UseBias)
                result.Add(new ParametersAndGradients() { Parameters = Bias, Gradients = BiasGradient });

            return result;
        }

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

        private static Shape GetOutShape(Shape inputShape, int filterWidth, int filterHeight, int stride, int filtersNum)
        {
            return new Shape((int)Math.Floor((float)(inputShape.Width - filterWidth) / stride + 1), (int)Math.Floor((float)(inputShape.Height - filterHeight) / stride + 1), filtersNum);
        }

        public Tensor Kernels;
        public Tensor Bias;
        public bool UseBias = true;

        public Tensor KernelsGradient;
        public Tensor BiasGradient;

        public Initializers.InitializerBase KernelInitializer = new Initializers.GlorotUniform();
        public Initializers.InitializerBase BiasInitializer = new Initializers.Zeros();

        public int FiltersNum;
        public int FilterSize;
        public int Stride;
    }
}

