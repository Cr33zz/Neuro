namespace Neuro
{
    public class Zeros : InitializerBase
    {
        public override Tensor Init(int[] shape, string name)
        {
            using (Backend.WithScope(name + "zeros"))
            {
                return Backend.Zeros(shape, name);
            }
        }
    }
}
