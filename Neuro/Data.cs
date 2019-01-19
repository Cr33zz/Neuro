using System;
using Neuro.Tensors;

namespace Neuro
{
    public class Data
    {
        public Data(Tensor[] inputs, Tensor[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public Data(Tensor input, Tensor output)
        {
            Inputs = new[] { input };
            Outputs = new[] { output };
        }

        public readonly Tensor[] Inputs;
        public readonly Tensor[] Outputs;

        public Tensor Input { get { return Inputs[0]; }}
        public Tensor Output { get { return Outputs[0]; } }
    }
}
