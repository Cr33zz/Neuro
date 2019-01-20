using System;
using System.Diagnostics;
using Neuro.Tensors;

namespace Neuro.Layers
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    public class Pooling : LayerBase
    {
        public Pooling(LayerBase prevLayer, int filterSize, int stride = 1, Tensor.PoolType type = Tensor.PoolType.Max)
            : base(prevLayer, Pooling.GetOutShape(prevLayer.OutputShape, filterSize, filterSize, stride))
        {
            Type = type;
            FilterSize = filterSize;
            Stride = stride;
        }

        // Use this constructor for input layer only!
        public Pooling(Shape inputShape, int filterSize, int stride = 1, Tensor.PoolType type = Tensor.PoolType.Max)
            : base(inputShape, Pooling.GetOutShape(inputShape, filterSize, filterSize, stride))
        {
            Type = type;
            FilterSize = filterSize;
            Stride = stride;
        }

        public override LayerBase Clone()
        {
            return new Pooling(InputShapes[0], FilterSize, Stride, Type);
        }

        protected override void FeedForwardInternal()
        {
            Inputs[0].Pool(FilterSize, Stride, Type, Tensor.PaddingType.Valid, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            Tensor.PoolGradient(Output, Inputs[0], outputGradient, FilterSize, Stride, Type, Tensor.PaddingType.Valid, InputsGradient[0]);
        }

        private static Shape GetOutShape(Shape inputShape, int filterWidth, int filterHeight, int stride)
        {
            return new Shape((int)Math.Floor((float)(inputShape.Width - filterWidth) / stride + 1),
                             (int)Math.Floor((float)(inputShape.Height - filterHeight) / stride + 1),
                             inputShape.Depth);
        }

        private readonly Tensor.PoolType Type;
        private readonly int FilterSize;
        private readonly int Stride;
    }
}
