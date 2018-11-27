using System.Diagnostics;
using System.Xml;
using Neuro.Tensors;
using System;
using System.Collections.Generic;

namespace Neuro.Layers
{
    public abstract class LayerBase
    {
        public Tensor Input { get; set; }
        public Tensor InputGradient { get; set; }
        public Tensor Output { get; protected set; }
        public Shape InputShape { get; }
        public Shape OutputShape { get; }
        public ActivationFunc Activation { get; }

        public virtual int GetParamsNum() { return 0; }

        // For serialization purposes only
        internal LayerBase() {}

        public abstract LayerBase Clone();

        public virtual void CopyParametersTo(LayerBase target)
        {
            if (!InputShape.Equals(target.InputShape) || !OutputShape.Equals(target.OutputShape))
                throw new Exception("Cannot copy parameters between incompatible layers.");
        }

        // The concept of layer is that it is a 'blockbox' that supports feed forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: error gradients (for its outputs) -> |learning| -> error gradients (for predecessing layer outputs) and internal parameters deltas
        // These error gradients are always of the same size as respective outputs and are saying now much each output
        // contributed to the final error)
        protected LayerBase(LayerBase prevLayer, Shape outputShape, ActivationFunc activation = null)
            : this(prevLayer.OutputShape, outputShape, activation)
        {
        }

        protected LayerBase(Shape inputShape, Shape outputShape, ActivationFunc activation = null)
        {
            InputShape = inputShape;
            OutputShape = outputShape;
            Activation = activation;
        }

        public Tensor FeedForward(Tensor input)
        {
            Debug.Assert(input.Width == InputShape.Width && input.Height == InputShape.Height && input.Depth == InputShape.Depth);

            Input = new Tensor(input);

            var outShape = new Shape(OutputShape.Width, OutputShape.Height, OutputShape.Depth, input.BatchSize);
            if (Output == null || !Output.Shape.Equals(outShape))
                Output = new Tensor(outShape);

            FeedForwardInternal();

            if (Activation != null)
            {
                Activation.Compute(Output, Output);

                if (NeuralNetwork.DebugMode)
                    Trace.WriteLine($"Activation({Activation.GetType().Name}) output:\n{Output}\n");
            }

            return Output;
        }

        public Tensor BackProp(Tensor outputGradient)
        {
            var deltaShape = new Shape(InputShape.Width, InputShape.Height, InputShape.Depth, outputGradient.BatchSize);
            if (InputGradient == null || !InputGradient.Shape.Equals(deltaShape))
                InputGradient = new Tensor(deltaShape);

            // apply derivative of our activation function to the errors computed by previous layer
            if (Activation != null)
            {
                Activation.Derivative(Output, outputGradient, outputGradient);

                if (NeuralNetwork.DebugMode)
                    Trace.WriteLine($"Activation({Activation.GetType().Name}) errors gradient:\n{outputGradient}\n");
            }

            BackPropInternal(outputGradient);

            return InputGradient;
        }

        public virtual List<ParametersAndGradients> GetParametersAndGradients()
        {
            return new List<ParametersAndGradients>();
        }

        // Must be called after adding to layers in a network
        public virtual void Init() {}

        //public delegate void ActivationFunc(Tensor input, bool deriv, Tensor result);

        protected abstract void FeedForwardInternal();

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of our that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        protected abstract void BackPropInternal(Tensor outputGradient);

        internal virtual void SerializeParameters(XmlElement elem) {}
        internal virtual void DeserializeParameters(XmlElement elem) {}
    }
}
