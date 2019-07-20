using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;

namespace Neuro.Optimizers
{
    // Implementation based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
    public class Adam : OptimizerBase
    {
        public Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            LearningRate = tf.constant(lr, name: "learning_rate");
            Beta1 = tf.constant(lr, name: "beta_1");
            Beta2 = tf.constant(lr, name: "beta_2");
            Epsilon = tf.constant(lr, name: "epsilon");
        }

        public override List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss)
        {
            var updates = new List<Tensor>();

            using (tf.name_scope("Adam"))
            {
                var grads = tf.gradients(loss, parameters.ToArray());

                Tensor t = Iteration + 1;
                Tensor lr_t = tf.multiply(LearningRate, (tf.sqrt(1 - tf.pow(Beta2, t)) / (1 - tf.pow(Beta1, t))));

                var shapes = parameters.Select(p => p.shape);
                var ms = shapes.Select(s => tf.zeros(s, name: "ms")).ToArray();
                var vs = shapes.Select(s => tf.zeros(s, name: "vs")).ToArray();

                for (var i = 0; i < parameters.Count; i++)
                {
                    Tensor p = parameters[i];
                    Tensor g = grads[i];

                    using (tf.name_scope(p.name))
                    {
                        var m = ms[i];
                        var v = vs[i];
                        var m_t = (Beta1 * m) + (1 - Beta1) * g;
                        var v_t = (Beta2 * v) + (1 - Beta2) * tf.square(g);
                        var p_t = tf.sub(p, lr_t * m_t / (tf.sqrt(v_t) + Epsilon));

                        updates.Add(state_ops.assign(m , m_t));
                        updates.Add(state_ops.assign(v, v_t));
                        updates.Add(state_ops.assign(p, p_t));
                    }
                }
            }

            return updates;
        }

        public override string ToString()
        {
            return $"Adam(lr={LearningRate})";
            //return $"Adam(lr={LearningRate}, beta1={Beta1}, beta2={Beta2}, epsilon={Epsilon})";
        }

        public Tensor LearningRate { get; protected set; }
        public Tensor Beta1 { get; protected set; }
        public Tensor Beta2 { get; protected set; }
        public Tensor Epsilon { get; protected set; }
    }
}
