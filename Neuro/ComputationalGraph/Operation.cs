using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public abstract class Operation : NodeBase
    {
        protected Operation(NodeBase[] inputNodes)
        {
            InputNodes = inputNodes;

            foreach (var inputNode in inputNodes)
                inputNode.Consumers.Add(this);

            Graph.Default.Operations.Add(this);
        }

        public virtual Tensor Compute(Tensor[] inputs)
        {
            Inputs = inputs;
            return inputs[0].Add(inputs[1]);
        }

        public abstract Tensor[] ComputeGradient(Tensor grad);

        protected Tensor[] Inputs;
    }
}
