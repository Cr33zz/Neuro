using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;

namespace Neuro
{
    public class Trainer
    {
        public Trainer(List<Tensor> inputs, List<Tensor> outputs, List<Tensor> targets, List<Tensor> updates)
        {
            Inputs = inputs;
            Outputs = outputs;
            Targets = targets;
            UpdatesOps = updates.Select(x => x.op).ToArray();
        }

        public NumSharp.NDArray Train(List<Array> inputs, List<Array> outputs)
        {
            var session = tf.Session();

            var init = tf.global_variables_initializer();
            session.run(init);

            var feed_dict = new Hashtable();

            for (int i = 0; i < Inputs.Count; ++i)
                feed_dict.Add(Inputs[i], inputs[i]);

            for (int i = 0; i < Targets.Count; ++i)
                feed_dict.Add(Targets[i], outputs[i]);

            return session.run(control_flow_ops.group(UpdatesOps), feed_dict);
        }

        public List<Tensor> Inputs;
        public List<Tensor> Outputs;
        public List<Tensor> Targets;
        public Operation[] UpdatesOps;
    }
}
