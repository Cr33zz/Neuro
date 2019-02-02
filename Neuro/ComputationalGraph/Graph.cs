using System.Collections.Generic;

namespace Neuro.ComputationalGraph
{
    public class Graph
    {
        public Graph()
        {
            if (Default == null)
                SetAsDefault();
        }

        public void SetAsDefault()
        {
            Default = this;
        }

        internal List<Placeholder> Placeholders = new List<Placeholder>();
        internal List<Operation> Operations = new List<Operation>();
        internal List<Variable> Variables = new List<Variable>();

        public static Graph Default { get; private set; }
    }
}
