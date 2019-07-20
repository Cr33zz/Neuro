using System.Xml;
using Neuro.Initializers;
using Tensorflow;

namespace Neuro
{
	public class Dense : LayerBase
    {
        public Dense(LayerBase inputLayer, int outputs, ActivationFunc activation = null, bool useBias = true, InitializerBase weightsInitializer = null, InitializerBase biasInitializer = null)
            : base(inputLayer, activation)
        {
            UseBias = useBias;
            OutputsNum = outputs;
            WeightsInitializer = weightsInitializer;
            BiasInitializer = biasInitializer ?? new Zeros();
        }

        // This constructor exists only for cloning purposes
        protected Dense()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Dense();
        }

        //protected override void OnClone(LayerBase source)
        //{
        //    base.OnClone(source);

        //    //var sourceDense = source as Dense;
        //    //Weights = sourceDense.Weights?.Clone();
        //    //Bias = sourceDense.Bias?.Clone();
        //    //UseBias = sourceDense.UseBias;
        //}

        public override void CopyParametersTo(LayerBase target, float tau)
        {
            base.CopyParametersTo(target, tau);

            //var targetDense = target as Dense;
            //Weights.CopyTo(targetDense.Weights, tau);
            //Bias.CopyTo(targetDense.Bias, tau);
        }

        protected override void OnBuild()
        {
			base.OnBuild();

            Weights = AddTrainableParam(new[] { InputShape.Dims.Get(-1), OutputsNum }, "weights", WeightsInitializer);

            if (UseBias)
                Bias = AddTrainableParam(new[] { OutputsNum }, "bias", BiasInitializer);

            Output = tf.matmul(InputLayers[0].Output, Weights);
            Output = Bias != null ? tf.add(Output, Bias) : Output;
            OutputShape = new Shape(Output.shape);
        }

        public override int GetParamsNum()
        {
            return Weights.shape.Product() + Bias.shape.Product();
        }

        public Tensor Weights;
        public Tensor Bias;
        public bool UseBias;
        public int OutputsNum;
        public InitializerBase WeightsInitializer;
        public InitializerBase BiasInitializer;

        internal override void SerializeParameters(XmlElement elem)
        {
            base.SerializeParameters(elem);
            //Weights.Serialize(elem, "Weights");
            //Bias.Serialize(elem, "Bias");
        }

        internal override void DeserializeParameters(XmlElement elem)
        {
            base.DeserializeParameters(elem);
            //Weights.Deserialize(elem["Weights"]);
            //Bias.Deserialize(elem["Bias"]);
        }
    }
}
