namespace Neuro
{
    public class Shape
    {
        public Shape(params int[] dims)
        {
            Dims = dims;
            NDim = dims.Length;
        }

        public int[] Dims;
        public int NDim;
    }
}
