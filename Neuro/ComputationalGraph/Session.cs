using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public class Session
    {
        public Tensor Run(Operation operation, Dictionary<Placeholder, Tensor> feeds)
        {
            var nodes = BuildGraph(operation);

            foreach (var node in nodes)
            {
                if (node is Placeholder p)
                    node.Output = feeds[p];
                else if (node is Variable v)
                    node.Output = v.Value;
                else
                {
                    var op = node as Operation;
                    var inputs = op.InputNodes.Select(x => x.Output).ToArray();
                    node.Output = op.Compute(inputs);
                }
            }

            return operation.Output;
        }

        private List<NodeBase> BuildGraph(NodeBase startNode)
        {
            List<NodeBase> result = new List<NodeBase>();
            ProcessNode(startNode, result);
            return result;
        }

        private void ProcessNode(NodeBase node, List<NodeBase> nodes)
        {
            if (node is Operation op)
            {
                foreach (var inputNode in op.InputNodes)
                    ProcessNode(inputNode, nodes);
            }
            nodes.Add(node);
        }
    }
}
