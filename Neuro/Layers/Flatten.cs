using Tensorflow;
using System.Linq;

namespace Neuro
{
    public class Flatten : LayerBase
    {
        public Flatten(LayerBase inputLayer)
            : base(inputLayer)
        {
        }

        protected Flatten()
        {
        }

        protected override void OnBuild()
        {
            base.OnBuild();

            var num_features = InputLayers[0].Output.shape.Skip(1).Product();
            OutputShape = new Shape(new[] { -1, num_features });
            Output = tf.reshape(InputLayers[0].Output, OutputShape.Dims);
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Flatten();
        }
    }
}
