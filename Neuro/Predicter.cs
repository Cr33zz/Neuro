using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace Neuro
{
    public class Predicter
    {
        public Predicter(List<Tensor> inputs, List<Tensor> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public List<Array> Predict(List<Array> inputs)
        {
            var session = Backend.Session;

            //var init = Backend.Graph.GetGlobalVariablesInitializer();
            //foreach (var op in init)
            //    session.Run(new TFOutput[0], new TFTensor[0], new TFOutput[0], new[] { op });

            var runner = session.GetRunner();

            foreach (var o in Outputs)
                runner.Fetch(o.Output);

            for (int i = 0; i < Inputs.Count; ++i)
                runner.AddInput(Inputs[i].Output, inputs[i]);

            var updated = runner.Run();
            return updated.Select(x => (Array)x.GetValue()).ToList();
        }

        public List<Tensor> Inputs;
        public List<Tensor> Outputs;
    }
}
