namespace Neuro
{
    public class Flatten : LayerBase
    {
        public Flatten(LayerBase inputLayer)
            : base(inputLayer)
        {
            Output = Backend.BatchFlatten(inputLayer.Output);
            OutputShape = new Shape(Output.Shape.ToIntArray());
        }

        protected Flatten()
        {
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Flatten();
        }
    }
}
