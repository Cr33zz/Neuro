using System.Collections.Generic;
using TensorFlow;

namespace Neuro.Optimizers
{
    public class SGD : OptimizerBase
    {
        public SGD(float lr = 0.01f)
        {
            LearningRate = Backend.Const(lr, "learning_rate");
        }

        public override List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss)
        {
            var updates = new List<Tensor>();

            using (Backend.WithScope("SGD"))
            {
                var grads = Backend.Gradients(loss, parameters);

                //updates.Add(Backend.AssignAdd(Iteration, 1.0f));

                for (var i = 0; i < parameters.Count; i++)
                {
                    Tensor p = parameters[i];
                    Tensor g = grads[i];

                    using (Backend.WithScope(p.Name))
                    {
                        Backend.Print(g);
                        updates.Add(Backend.Assign(p, p - LearningRate * g));
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
