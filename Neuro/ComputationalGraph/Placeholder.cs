using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neuro.Tensors;

namespace Neuro.ComputationalGraph
{
    public class Placeholder : NodeBase
    {
        public Placeholder(Shape shape)
        {
            Shape = shape;
            Graph.Default.Placeholders.Add(this);
        }

        private Shape Shape;
    }
}
