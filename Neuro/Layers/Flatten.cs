using System.Diagnostics;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Flatten : LayerBase
    {
        public Flatten(LayerBase prevLayer)
            : base(prevLayer.OutputShape, new Shape(1, prevLayer.OutputShape.Length))
        {
        }

        protected override void FeedForwardInternal()
        {
            // output is already of proper shape thanks to LayerBase.FeedForward
            Input.CopyTo(Output);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Flatten() output:\n{Output}\n");
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            InputGradient = outputGradient.Reshaped(Input.Shape);

            if (NeuralNetwork.DebugMode)
                Trace.WriteLine($"Flatten() errors gradient:\n{InputGradient}\n");
        }
    }
}
