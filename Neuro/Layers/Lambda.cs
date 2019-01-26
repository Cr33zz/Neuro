using Neuro.Tensors;

namespace Neuro.Layers
{
    // This layer allows user to implement custom inputs mixing/processing
    abstract class Lambda : LayerBase
    {
        public Lambda(LayerBase[] inputLayers, Shape outputShape)
            : base(inputLayers, outputShape)
        {
        }

        // This constructor should only be used for input layer
        public Lambda(Shape[] inputShapes, Shape outputShape)
            : base(inputShapes, outputShape)
        {
        }

        protected override void FeedForwardInternal()
        {
            ProcessInputs(Inputs, Output);
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {
            BackPropOutputGradient(outputGradient, InputsGradient);
        }

        // Implement your custom algorithm for generating output from inputs
        protected abstract void ProcessInputs(Tensor[] inputs, Tensor output);

        // In this function you need to say how each input was responsible for output error (gradient)
        // For example: if you simply sum all inputs to produce output (each input has weight 1) then each input is equally responsible for error
        // and gradient for each input should be the same as output gradient; in case of average weight for each input would be 1/number_of_inputs.
        protected abstract void BackPropOutputGradient(Tensor outputGradient, Tensor[] inputsGradient);
    }
}
