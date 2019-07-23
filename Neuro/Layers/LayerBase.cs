using System.Diagnostics;
using System.Xml;
using Neuro.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace Neuro.Layers
{
    public abstract class LayerBase
    {
        public Shape[] InputShapes { get; protected set; }
        public Shape InputShape { get {return InputShapes[0];} }
        public Tensor[] Inputs { get; set; }
        public Tensor Input { get { return Inputs[0]; } }
        public Tensor[] InputsGradient { get; set; }
        public Tensor InputGradient { get { return InputsGradient[0]; } }
        public Tensor Output { get; protected set; }
        public Shape OutputShape { get; private set; }
        public ActivationFunc Activation { get; private set; }
        public string Name { get; set; }
        internal List<LayerBase> InputLayers = new List<LayerBase>();
        internal List<LayerBase> OutputLayers = new List<LayerBase>();

        // The concept of layer is that it is a 'block box' that supports feed forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: error gradients (for its outputs) -> |learning| -> error gradients (for predecessing layer outputs) and internal parameters deltas
        // These error gradients are always of the same size as respective outputs and are saying now much each output
        // contributed to the final error)
        protected LayerBase(LayerBase inputLayer, Shape outputShape, ActivationFunc activation = null)
            : this(new [] {inputLayer.OutputShape}, outputShape, activation)
        {
            InputLayers.Add(inputLayer);
            inputLayer.OutputLayers.Add(this);
        }

        protected LayerBase(LayerBase[] inputLayers, Shape outputShape, ActivationFunc activation = null)
            : this(inputLayers.Select(l => l.OutputShape).ToArray(), outputShape, activation)
        {
            InputLayers.AddRange(inputLayers);
            foreach (var inLayer in inputLayers)
                inLayer.OutputLayers.Add(this);
        }

        // This constructor should only be used for input layer
        protected LayerBase(Shape inputShape, Shape outputShape, ActivationFunc activation = null)
            : this(new[] { inputShape }, outputShape, activation)
        {
        }

        // This constructor should only be used for input layer
        protected LayerBase(Shape[] inputShapes, Shape outputShape, ActivationFunc activation = null)
        {
            InputShapes = inputShapes;
            OutputShape = outputShape;
            Activation = activation;
            Name = GenerateName();
        }

        // This constructor exists only for cloning purposes
        protected LayerBase()
        {
        }

        protected abstract LayerBase GetCloneInstance();

        public LayerBase Clone()
        {
			Init(); // make sure parameter matrices are created
            var clone = GetCloneInstance();
            clone.OnClone(this);
            return clone;
        }

        protected virtual void OnClone(LayerBase source)
        {
            InputShapes = source.InputShapes;
            OutputShape = source.OutputShape;
            Activation = source.Activation;
            Name = source.Name;
            //Initialized = source.Initialized;
        }

        public virtual void CopyParametersTo(LayerBase target, float tau = float.NaN)
        {
            if (!InputShapes.Equals(target.InputShapes) || !OutputShape.Equals(target.OutputShape))
                throw new Exception("Cannot copy parameters between incompatible layers.");
        }

        public Tensor FeedForward(Tensor input)
        {
            return FeedForward(new []{input});
        }

        public Tensor FeedForward(Tensor[] inputs)
        {
            if (!Initialized)
                Init();

            //Debug.Assert(input.Width == InputShape.Width && input.Height == InputShape.Height && input.Depth == InputShape.Depth);

            Inputs = inputs;

            var outShape = new Shape(OutputShape.Width, OutputShape.Height, OutputShape.Depth, inputs[0].BatchSize);
            // shape comparison is required for cases when last batch has different size
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

        public Tensor[] BackProp(Tensor outputGradient)
        {
            if (InputsGradient == null)
                InputsGradient = new Tensor[InputShapes.Length];

            for (int i = 0; i < InputShapes.Length; ++i)
            {
                var inputShape = InputShapes[i];
                var deltaShape = new Shape(inputShape.Width, inputShape.Height, inputShape.Depth, outputGradient.BatchSize);
                if (InputsGradient[i] == null || !InputsGradient[i].Shape.Equals(deltaShape))
                    InputsGradient[i] = new Tensor(deltaShape);
            }

            // apply derivative of our activation function to the errors computed by previous layer
            if (Activation != null)
            {
                Activation.Derivative(Output, outputGradient, outputGradient);

                if (NeuralNetwork.DebugMode)
                    Trace.WriteLine($"Activation({Activation.GetType().Name}) errors gradient:\n{outputGradient}\n");
            }

            BackPropInternal(outputGradient);

            return InputsGradient;
        }

        public virtual int GetParamsNum()
        {
            return 0;
        }

        public virtual List<ParametersAndGradients> GetParametersAndGradients()
        {
            return new List<ParametersAndGradients>();
        }

        public void Init()
        {
			if (Initialized)
				return;

			OnInit();
			Initialized = true;
        }

        protected virtual void OnInit()
        {
        }

        private bool Initialized = false;

        //public delegate void ActivationFunc(Tensor input, bool deriv, Tensor result);

        protected abstract void FeedForwardInternal();

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of our that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        protected abstract void BackPropInternal(Tensor outputGradient);

        internal virtual void SerializeParameters(XmlElement elem) {}
        internal virtual void DeserializeParameters(XmlElement elem) {}

        private string GenerateName()
        {
            if (!LayersCountPerType.ContainsKey(GetType()))
                LayersCountPerType.Add(GetType(), 0);
            return $"{GetType().Name.ToLower()}_{++LayersCountPerType[GetType()]}";
        }

        private static Dictionary<Type, int> LayersCountPerType = new Dictionary<Type, int>();
    }
}
