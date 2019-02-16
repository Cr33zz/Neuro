using System;
using System.Diagnostics;

namespace Neuro.Layers
{
    // https://www.youtube.com/watch?v=8oOgPUO-TBY
    /*public class Pooling : LayerBase
    {
        public Pooling(LayerBase inputLayer, int filterSize, int stride = 1, TFTensor.PoolType type = TFTensor.PoolType.Max)
            : base(inputLayer, Pooling.GetOutShape(inputLayer.OutputShape, filterSize, filterSize, stride))
        {
            Type = type;
            FilterSize = filterSize;
            Stride = stride;
        }

        // Use this constructor for input layer only!
        public Pooling(TFShape inputShape, int filterSize, int stride = 1, TFTensor.PoolType type = TFTensor.PoolType.Max)
            : base(inputShape, Pooling.GetOutShape(inputShape, filterSize, filterSize, stride))
        {
            Type = type;
            FilterSize = filterSize;
            Stride = stride;
        }

        protected Pooling()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Pooling();
        }

        protected override void OnClone(LayerBase source)
        {
            base.OnClone(source);

            var sourcePool = source as Pooling;
            Type = sourcePool.Type;
            FilterSize = sourcePool.FilterSize;
            Stride = sourcePool.Stride;
        }

        protected override void FeedForwardInternal()
        {
            Inputs[0].Pool(FilterSize, Stride, Type, TFTensor.PaddingType.Valid, Output);
        }

        protected override void BackPropInternal(TFTensor outputGradient)
        {
            TFTensor.PoolGradient(Output, Inputs[0], outputGradient, FilterSize, Stride, Type, TFTensor.PaddingType.Valid, InputsGradient[0]);
        }

        private static TFShape GetOutShape(TFShape inputShape, int filterWidth, int filterHeight, int stride)
        {
            return new TFShape((int)Math.Floor((float)(inputShape.Width - filterWidth) / stride + 1),
                             (int)Math.Floor((float)(inputShape.Height - filterHeight) / stride + 1),
                             inputShape.Depth);
        }

        private TFTensor.PoolType Type;
        private int FilterSize;
        private int Stride;
    }*/
}
