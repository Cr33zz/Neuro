using Tensorflow;

namespace Neuro
{
    public class Zeros : InitializerBase
    {
        public override Tensor Init(int[] shape, string name)
        {
            using (tf.name_scope(name + "zeros"))
            {
                return tf.zeros(shape, name: name);
            }
        }
    }
}
