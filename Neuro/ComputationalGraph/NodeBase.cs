using System.Collections.Generic;
using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public abstract class NodeBase
    {
        public string Name { get; protected set; }
        internal List<NodeBase> Consumers = new List<NodeBase>();
        internal Tensor Output;
    }
}
