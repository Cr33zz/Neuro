namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public Array Reshape(params int[] dims)
            {
                Storage.Reshape(dims);
                return this;
            }
        }
    }
}
