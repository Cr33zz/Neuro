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

            Output = Backend.BatchFlatten(InputLayers[0].Output);
            OutputShape = new Shape(Output.Shape.ToIntArray());
        }

        protected override LayerBase GetCloneInstance()
        {
            return new Flatten();
        }
    }
}
