using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;

namespace Neuro
{
    public class Predicter
    {
        public Predicter(List<Tensor> inputs, List<Tensor> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public NumSharp.NDArray Predict(List<Array> inputs)
        {
            var session = tf.Session();

            //var init = tf.Graph.GetGlobalVariablesInitializer();
            //foreach (var op in init)
            //    session.Run(new Tensorflow.TF_Output[0], new Tensorflow.Tensor[0], new Tensorflow.TF_Output[0], new[] { op });

            var feed_dict = new Hashtable();

            for (int i = 0; i < Inputs.Count; ++i)
                feed_dict.Add(Inputs[i], inputs[i]);

            return session.run(control_flow_ops.group(Outputs.Select(x => x.op).ToArray()), feed_dict);
        }

        public List<Tensor> Inputs;
        public List<Tensor> Outputs;
    }
}
