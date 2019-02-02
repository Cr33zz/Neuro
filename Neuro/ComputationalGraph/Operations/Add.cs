using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public static partial class Ops
    {
        private class Add : Operation
        {
            public Add(NodeBase a, NodeBase b)
                : base(new[] {a, b})
            {
            }

            public override Tensor Compute(Tensor[] inputs)
            {
                base.Compute(inputs);
                return inputs[0].Add(inputs[1]);
            }
        }

        public static Operation add(NodeBase a, NodeBase b)
        {
            return new Add(a, b);
        }
    }
}
