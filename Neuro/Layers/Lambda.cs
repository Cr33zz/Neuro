﻿namespace Neuro.Layers
{
    // This layer allows user to implement custom inputs mixing/processing
    //public class Lambda : LayerBase
    //{
    //    // Implement your custom algorithm for generating output from inputs
    //    public delegate void LambdaFunc(Tensorflow.Tensor[] inputs, Tensorflow.Tensor output);

    //    // In this function you need to say how each input was responsible for output error (gradient)
    //    // For example: if you simply sum all inputs to produce output (each input has weight 1) then each input is equally responsible for error
    //    // and gradient for each input should be the same as output gradient; in case of average weight for each input would be 1/number_of_inputs.
    //    public delegate void LambdaBackpropFunc(Tensorflow.Tensor outputGradient, Tensorflow.Tensor[] inputsGradient);

    //    public Lambda(LayerBase[] inputLayers, TFShape outputShape, LambdaFunc processInputsFunc, LambdaBackpropFunc backPropOutputGradientFunc, ActivationFunc activation = null)
    //        : base(inputLayers, outputShape, activation)
    //    {
    //        ProcessInputsFunc = processInputsFunc;
    //        BackPropOutputGradientFunc = backPropOutputGradientFunc;
    //    }

    //    // This constructor should only be used for input layer
    //    public Lambda(TFShape[] inputShapes, TFShape outputShape, LambdaFunc processInputsFunc, LambdaBackpropFunc backPropOutputGradientFunc, ActivationFunc activation = null)
    //        : base(inputShapes, outputShape, activation)
    //    {
    //        ProcessInputsFunc = processInputsFunc;
    //        BackPropOutputGradientFunc = backPropOutputGradientFunc;
    //    }

    //    protected Lambda()
    //    {
    //    }

    //    protected override LayerBase GetCloneInstance()
    //    {
    //        return new Lambda();
    //    }

    //    protected override void OnClone(LayerBase source)
    //    {
    //        base.OnClone(source);

    //        var sourceLambda = source as Lambda;
    //        ProcessInputsFunc = sourceLambda.ProcessInputsFunc;
    //        BackPropOutputGradientFunc = sourceLambda.BackPropOutputGradientFunc;
    //    }

    //    protected override void FeedForwardInternal()
    //    {
    //        ProcessInputsFunc(Inputs, Output);
    //    }

    //    protected override void BackPropInternal(Tensorflow.Tensor outputGradient)
    //    {
    //        BackPropOutputGradientFunc(outputGradient, InputsGradient);
    //    }

    //    private LambdaFunc ProcessInputsFunc;
    //    private LambdaBackpropFunc BackPropOutputGradientFunc;
    //}
}
