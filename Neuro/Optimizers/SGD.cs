using System.Collections.Generic;
using Tensorflow;

namespace Neuro.Optimizers
{
    public class SGD : OptimizerBase
    {
        public SGD(float lr = 0.01f)
        {
            LearningRate = tf.constant(lr, name: "learning_rate");
        }

        public override List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss)
        {
            var updates = new List<Tensor>();

            using (tf.name_scope("SGD"))
            {
                var grads = tf.gradients(loss, parameters.ToArray());

                //updates.Add(tf.AssignAdd(Iteration, 1.0f));

                for (var i = 0; i < parameters.Count; i++)
                {
                    Tensor p = parameters[i];
                    Tensor g = grads[i];

                    using (tf.name_scope(p.name))
                    {
                        //tf.Print(g);
                        updates.Add(state_ops.assign(p, p - LearningRate * g));
                    }
                }
            }

            return updates;
        }

        public override string ToString()
        {
            return $"SGD(lr={LearningRate})";
        }

        public Tensor LearningRate { get; protected set; }
    }
}
