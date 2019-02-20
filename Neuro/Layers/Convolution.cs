using System;
using System.Diagnostics;
using System.Xml;
using Neuro.Initializers;
using TensorFlow;

namespace Neuro
{
    public class Convolution : LayerBase
    {
        public Convolution(LayerBase inputLayer, int filterSize, int filtersNum, int stride, ActivationFunc activation = null, InitializerBase kernelInitializer = null, InitializerBase biasInitializer = null)
            : base(inputLayer)
        {
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer ?? new Zeros();
            FilterSize = filterSize;
            FiltersNum = filtersNum;
            Stride = stride;
        }

        protected Convolution()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Convolution();
        }

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target, tau);

            //var targetConv = target as Convolution;
            //Kernels.CopyTo(targetConv.Kernels, tau);
            //Bias.CopyTo(targetConv.Bias, tau);
        }

        protected override void OnBuild()
        {
			base.OnBuild();

            Kernels = AddTrainableParam(new[] { FilterSize, FilterSize, InputShape.Dims.Get(-1), FiltersNum }, "kernels", KernelInitializer);
            Bias = AddTrainableParam(new[] { FiltersNum }, "bias", BiasInitializer);

            Output = Backend.Add(Backend.Conv2D(InputLayers[0].Output, Kernels, new[] { Stride, Stride }, Backend.PaddingType.Valid), Bias);
            OutputShape = new Shape(Output.Shape.ToIntArray());
        }

        public override int GetParamsNum()
        {
            return Kernels.Shape.ToIntArray().Product() + Bias.Shape.ToIntArray().Product();
        }

		internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            //Kernels.Serialize(elem, "Kernels");
            //Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            //Kernels.Deserialize(elem["Kernels"]);
            //Bias.Deserialize(elem["Bias"]);
        }

        public Tensor Kernels;
        public Tensor Bias;
        public int FilterSize;
        public int Stride;
        public int FiltersNum;
        public InitializerBase KernelInitializer;
        public InitializerBase BiasInitializer;
    }
}

