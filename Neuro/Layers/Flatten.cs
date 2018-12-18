using System.Diagnostics;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Flatten : LayerBase
    {
        public Flatten(LayerBase prevLayer)
            : this(prevLayer.OutputShape)
        {
        }

        public Flatten(Shape inputShape)
            : base(inputShape, new Shape(1, inputShape.Length))
        {
        }

        public override LayerBase Clone()
        {
            return new Flatten(InputShapes[0]);
        }

        protected override void FeedForwardInternal()
        {
            // output is already of proper shape thanks to LayerBase.FeedForward
            Inputs[0].CopyTo(Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            InputsGradient[0] = outputGradient.Reshaped(Inputs[0].Shape);
        }
    }
}
