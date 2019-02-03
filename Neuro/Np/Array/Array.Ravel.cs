namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public Array Ravel()
            {
                Storage.Reshape(Storage.Shape.Size);
                return this;
            }
        }
    }
}
