using System.Collections.Generic;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Models
{
    public abstract class ModelBase
    {
        public abstract ModelBase Clone();
        public abstract void FeedForward(Tensor[] inputs);
        public abstract void BackProp(Tensor[] deltas);
        public virtual void Optimize() { }
        public abstract IEnumerable<LayerBase> GetLayers();
        public abstract Tensor[] GetOutputs();
        public abstract IEnumerable<LayerBase> GetOutputLayers();
        public abstract int GetOutputLayersCount();
        public virtual string Summary() { return ""; }
        public virtual void SaveStateXml(string filename) { }
        public virtual void LoadStateXml(string filename) { }

        public LayerBase GetLayer(string name)
        {
            foreach (var layer in GetLayers())
                if (layer.Name == name)
                    return layer;
            return null;
        }

        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            var result = new List<ParametersAndGradients>();

            foreach (var layer in GetLayers())
            {
                result.AddRange(layer.GetParametersAndGradients());
            }

            return result;
        }
    }
}
