namespace Neuro.Layers
{
    //public class Merge : LayerBase
    //{
    //    public enum Mode
    //    {
    //        Sum,
    //        Avg,
    //        Max,
    //        Min
    //    }

    //    public Merge(LayerBase[] inputLayers, Mode mergeMode, ActivationFunc activation = null)
    //        : base(inputLayers, inputLayers[0].OutputShape, activation)
    //    {
    //        MergeMode = mergeMode;
    //    }

    //    // This constructor should only be used for input layer
    //    public Merge(TFShape[] inputShapes, Mode mergeMode, ActivationFunc activation = null)
    //        : base(inputShapes, inputShapes[0], activation)
    //    {
    //        MergeMode = mergeMode;
    //    }

    //    protected Merge()
    //    {
    //    }

    //    protected override LayerBase GetCloneInstance()
    //    {
    //        return new Merge();
    //    }

    //    protected override void OnClone(LayerBase source)
    //    {
    //        base.OnClone(source);

    //        var sourceMerge = source as Merge;
    //        MergeMode = sourceMerge.MergeMode;
    //    }

    //    protected override void FeedForwardInternal()
    //    {
    //        switch (MergeMode)
    //        {
    //            case Mode.Avg:
    //                TFTensor.MergeAvg(Inputs, Output);
    //                break;
    //            case Mode.Max:
    //                TFTensor.MergeMax(Inputs, Output);
    //                break;
    //            case Mode.Min:
    //                TFTensor.MergeMin(Inputs, Output);
    //                break;
    //            case Mode.Sum:
    //                TFTensor.MergeSum(Inputs, Output);
    //                break;
    //        }
    //    }

    //    protected override void BackPropInternal(TFTensor outputGradient)
    //    {
    //        switch (MergeMode)
    //        {
    //            case Mode.Avg:
    //                TFTensor.MergeAvgGradient(Output, Inputs, outputGradient, InputsGradient);
    //                break;
    //            case Mode.Max:
    //            case Mode.Min:
    //                TFTensor.MergeMinMaxGradient(Output, Inputs, outputGradient, InputsGradient);
    //                break;
    //            case Mode.Sum:
    //                TFTensor.MergeSumGradient(Output, Inputs, outputGradient, InputsGradient);
    //                break;
    //        }
    //    }

    //    public Mode MergeMode { get; private set; }
    //}
}
