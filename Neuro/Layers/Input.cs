namespace Neuro
{
    public class Input : LayerBase
    {
        public Input(int[] shape)
            : base(new Shape(shape))
        {
        }

        protected Input()
        {
        }

        protected override void OnBuild()
        {
            base.OnBuild();

            Inputs = new[] { Backend.Placeholder(InputShape.Dims) };
            Output = Backend.Identity(Input);
            OutputShape = new Shape(Output.Shape.ToIntArray());
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Input();
        }
    }
}
