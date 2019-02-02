using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public class Variable : NodeBase
    {
        public Variable(Tensor initValue)
        {
            Value = initValue;
            Graph.Default.Variables.Add(this);
        }

        public Tensor Value { get; private set; }
    }
}
