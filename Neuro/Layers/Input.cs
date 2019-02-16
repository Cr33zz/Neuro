namespace Neuro
{
    public class Input : LayerBase
    {
        public Input(int[] shape)
            : base(new Shape(shape))
        {
	        Output = Backend.Placeholder(shape);
        }

        protected Input()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Input();
        }
    }
}
