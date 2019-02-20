using System.Diagnostics;
using System.Xml;
using System;
using System.Collections.Generic;
using System.Linq;
using Neuro.Initializers;
using TensorFlow;

namespace Neuro
{
    public abstract class LayerBase
    {
        public Shape[] InputShapes { get; protected set; }
        public Shape InputShape  => InputShapes[0];
        public Tensor[] Inputs { get; set; }
        public Tensor Input => Inputs[0];
        public Shape OutputShape { get; protected set; }
        public Tensor Output;
        public ActivationFunc Activation { get; private set; }
        public string Name { get; set; }

        protected LayerBase(LayerBase inputLayer, ActivationFunc activation = null)
            : this(new [] {inputLayer.OutputShape}, activation)
        {
            InputLayers.Add(inputLayer);
            inputLayer.OutputLayers.Add(this);
		}

        protected LayerBase(LayerBase[] inputLayers, ActivationFunc activation = null)
            : this(inputLayers.Select(l => l.OutputShape).ToArray(), activation)
        {
            InputLayers.AddRange(inputLayers);
            foreach (var inLayer in inputLayers)
                inLayer.OutputLayers.Add(this);
		}

        // This constructor should only be used for input layer
        protected LayerBase(Shape inputShape, ActivationFunc activation = null)
            : this(new[] { inputShape }, activation)
        {
        }

        // This constructor should only be used for input layer
        protected LayerBase(Shape[] inputShapes, ActivationFunc activation = null)
        {
            InputShapes = inputShapes;
            Activation = activation;
            Name = GenerateName();
        }

        // This constructor exists only for cloning purposes
        protected LayerBase()
        {
        }

        protected abstract LayerBase GetCloneInstance();

   //     public LayerBase Clone()
   //     {
			//Init(); // make sure parameter matrices are created
   //         var clone = GetCloneInstance();
   //         clone.OnClone(this);
   //         return clone;
   //     }

   //     protected virtual void OnClone(LayerBase source)
   //     {
   //         InputShapes = source.InputShapes;
   //         OutputShape = source.OutputShape;
   //         Activation = source.Activation;
   //         Name = source.Name;
   //         Initialized = source.Initialized;
   //     }

        public virtual void CopyParametersTo(LayerBase target, float tau = float.NaN)
        {
            //if (!InputShapes.Equals(target.InputShapes) || !OutputShape.Equals(target.OutputShape))
            //    throw new Exception("Cannot copy parameters between incompatible layers.");
        }

        public virtual int GetParamsNum()
        {
            return 0;
        }

        protected Tensor AddTrainableParam(int[] shape, string name, InitializerBase initializer = null)
        {
            if (initializer == null)
                initializer = new GlorotUniform();

            var param = Backend.Variable(initializer.Init(shape, name), name);
            TrainableParams.Add(param);
            return param;
        }

        public void Build()
        {
            if (Built)
                return;

            using (Backend.WithScope(Name))
            {
                if (InputLayers.Count > 0)
                    InputShapes = InputLayers.Select(x => x.OutputShape).ToArray();

                OnBuild();

                if (Activation != null)
                    Output = Activation.Build(Output);
            }

            Built = true;
        }

        protected virtual void OnBuild()
        {
        }

        private bool Built;

        internal virtual void SerializeParameters(XmlElement elem) {}
        internal virtual void DeserializeParameters(XmlElement elem) {}

        public List<Tensor> TrainableParams = new List<Tensor>();
        internal List<LayerBase> InputLayers = new List<LayerBase>();
        internal List<LayerBase> OutputLayers = new List<LayerBase>();        

        private string GenerateName()
        {
            if (!LayersCountPerType.ContainsKey(GetType()))
                LayersCountPerType.Add(GetType(), 0);
            return $"{GetType().Name.ToLower()}_{++LayersCountPerType[GetType()]}";
        }

        private static Dictionary<Type, int> LayersCountPerType = new Dictionary<Type, int>();
    }
}
