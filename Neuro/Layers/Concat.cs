using System;
using System.Linq;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Concat : LayerBase
    {
        public Concat(LayerBase[] inputLayers)
            : base(inputLayers, new Shape(1, inputLayers.Select(x => x.OutputShape.Length).Sum()))
        {
        }

        public override LayerBase Clone()
        {
            throw new NotImplementedException();
        }

        protected override void FeedForwardInternal()
        {
            // output is already of proper shape thanks to LayerBase.FeedForward
            Tensor.Concat(Inputs, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            outputGradient.Split(InputsGradient);
        }
    }
}
