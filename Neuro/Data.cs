using System;
using System.Collections.Generic;
using TensorFlow;

namespace Neuro
{
    public class Data
    {
        public Data(Array[] inputs, Array[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public Data(Array input, Array output)
        {
            Inputs = new[] { input };
            Outputs = new[] { output };
        }

        public readonly Array[] Inputs;
        public readonly Array[] Outputs;

        public Array Input { get { return Inputs[0]; }}
        public Array Output { get { return Outputs[0]; } }
    }
}
