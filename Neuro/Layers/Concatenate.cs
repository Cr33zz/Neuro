using System;
using System.Linq;

namespace Neuro.Layers
{
    //public class Concatenate : LayerBase
    //{
    //    public Concatenate(LayerBase[] inputLayers, ActivationFunc activation = null)
    //        : base(inputLayers, new TFShape(1, inputLayers.Select(x => x.OutputShape.Length).Sum()))
    //    {
    //    }

    //    protected Concatenate()
    //    {
    //    }

    //    protected override LayerBase GetCloneInstance()
    //    {
    //        return new Concatenate();
    //    }

    //    protected override void FeedForwardInternal()
    //    {
    //        // output is already of proper shape thanks to LayerBase.FeedForward
    //        TFTensor.Concat(Inputs, Output);
    //    }

    //    protected override void BackPropInternal(TFTensor outputGradient)
    //    {
    //        outputGradient.Split(InputsGradient);
    //    }
    //}
}
