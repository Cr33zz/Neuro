using System.Xml;
using Neuro.Initializers;

namespace Neuro
{
	public class Dense : LayerBase
    {
        public Dense(LayerBase inputLayer, int outputs, ActivationFunc activation = null, bool useBias = true, InitializerBase weightsInitializer = null, InitializerBase biasInitializer = null)
            : base(inputLayer, activation)
        {
            Weights = AddTrainableParam(new[] { InputShape.Dims.Get(-1), outputs }, weightsInitializer);

            if (useBias)
                Bias = AddTrainableParam(new[] { outputs }, biasInitializer);
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

            var linear = Backend.Dot(InputLayers[0].Output, Weights);
            OutputShape = new Shape(linear.Shape.ToIntArray());

            Output = Bias != null ? Backend.Add(linear, Bias) : linear;
        }

        public override int GetParamsNum()
        {
            return Weights.Shape.ToIntArray().Product() + Bias.Shape.ToIntArray().Product();
        }

        public Tensor Weights;
        public Tensor Bias;

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
