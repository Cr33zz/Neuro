using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace Neuro
{
    public class Trainer
    {
        public Trainer(List<Tensor> inputs, List<Tensor> outputs, List<Tensor> targets, List<Tensor> updates)
        {
            Inputs = inputs;
            Outputs = outputs;
            Targets = targets;
            UpdatesOps = updates.Select(x => x.Output.Operation).ToList();
        }

        public List<Tensor> Train(List<Array> inputs, List<Array> outputs)
        {
            var session = Backend.Session;

            var init = Backend.Graph.GetGlobalVariablesInitializer();
            foreach (var op in init)
                session.Run(new TFOutput[0], new TFTensor[0], new TFOutput[0], new[] { op });

            var runner = session.GetRunner();

            foreach (var o in Outputs)
                runner.Fetch(o.Output);

            foreach (var op in UpdatesOps)
                runner.AddTarget(op);

            for (int i = 0; i < Inputs.Count; ++i)
                runner.AddInput(Inputs[i].Output, inputs[i]);

            for (int i = 0; i < Targets.Count; ++i)
                runner.AddInput(Targets[i].Output, outputs[i]);

            var updated = runner.Run();
            return updated.Select(x => new Tensor(x)).ToList();
        }

        public List<Tensor> Inputs;
        public List<Tensor> Outputs;
        public List<Tensor> Targets;
        public List<TFOperation> UpdatesOps;
    }
}
