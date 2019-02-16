using System.Collections.Generic;
using TensorFlow;

namespace Neuro.Optimizers
{
    public class SGD : OptimizerBase
    {
        public SGD(float lr = 0.01f)
        {
            LearningRate = new Tensor(lr);
        }

        protected override List<Tensor> GenerateUpdates(List<Tensor> parameters, Tensor loss)
        {
            using (Backend.WithScope("SGD"))
            {
                var grads = Backend.Gradients(loss, parameters);

                Updates.Add(Backend.AssignAdd(Iteration, 1.0f));

                for (var i = 0; i < parameters.Count; i++)
                {
                    Tensor p = parameters[i];
                    Tensor g = grads[i];

                    Updates.Add(Backend.Assign(p, p - g * LearningRate));
                }
            }

            return Updates;
        }

        public override string ToString()
        {
            return $"SGD(lr={LearningRate})";
        }

        public Tensor LearningRate { get; protected set; }
        private List<Tensor> Updates = new List<Tensor>();
    }
}
