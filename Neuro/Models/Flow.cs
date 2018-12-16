using System.Collections.Generic;
using System.Linq;
using Neuro.Layers;
using Neuro.Tensors;

namespace Neuro.Models
{
    public class Flow : ModelBase
    {
        public Flow(LayerBase[] inputLayers, LayerBase[] outputLayers)
        {
            InputLayers = inputLayers.ToList();
            OutputLayers = outputLayers.ToList();
        }

        public override void FeedForward(Tensor[] inputs)
        {
            for (int i = 0; i < InputLayers.Count; ++i)
                InputLayers[i].FeedForward(new[] {inputs[i]});

            foreach (var layer in Order)
            {
                // layers with no input layers have are already been fed forward
                if (layer.InputLayers.Count == 0)
                    continue;

                var ins = new Tensor[layer.InputLayers.Count];
                for (int i = 0; i < layer.InputLayers.Count; ++i)
                    ins[i] = layer.InputLayers[i].Output;

                layer.FeedForward(ins);
            }
        }

        public override void BackProp(Tensor[] deltas)
        {
            for (int i = 0; i < OutputLayers.Count; ++i)
                InputLayers[i].BackProp(deltas[i]);

            foreach (var layer in ReversedOrder)
            {
                // layers with no input layers have are already been fed forward
                if (layer.OutputLayers.Count == 0)
                    continue;

                Tensor avgDelta = new Tensor(layer.OutputShape);
                for (int i = 0; i < layer.OutputLayers.Count; ++i)
                {
                    // we need to find this layer index in output layer's inputs to grab proper delta
                    for (int j = 0; j < layer.OutputLayers[i].InputLayers.Count; ++j)
                    {
                        if (layer.OutputLayers[i].InputLayers[j] == layer)
                            avgDelta.Add(layer.OutputLayers[i].InputsGradient[j], avgDelta);
                    }
                }

                avgDelta.Div(layer.OutputLayers.Count, avgDelta);

                layer.BackProp(avgDelta);
            }
        }

        public override IEnumerable<LayerBase> GetOutputLayers()
        {
            return OutputLayers;
        }

        public override int GetOutputLayersCount()
        {
            return OutputLayers.Count;
        }

        public virtual void Optimize()
        {
            List<LayerBase> visited = new List<LayerBase>();

            foreach (var inputLayer in InputLayers)
                ProcessLayer(inputLayer, ref visited);

            ReversedOrder = new List<LayerBase>(Order);
            ReversedOrder.Reverse();
        }

        private void ProcessLayer(LayerBase layer, ref List<LayerBase> visited)
        {
            bool allInputLayersVisited = true;

            foreach (var inLayer in layer.InputLayers)
            {
                if (!visited.Contains(inLayer))
                {
                    allInputLayersVisited = false;
                    break;
                }
            }

            if (!allInputLayersVisited)
                return;

            Order.Add(layer);
            visited.Add(layer);

            foreach (var outLayer in layer.OutputLayers)
                ProcessLayer(outLayer, ref visited);
        }

        public override ModelBase Clone()
        {
            throw new NotImplementedException();
        }

        public override IEnumerable<LayerBase> GetLayers()
        {
            return Order;
        }

        public override Tensor[] GetOutputs()
        {
            Tensor[] outputs = new Tensor[OutputLayers.Count];
            for (int i = 0; i < OutputLayers.Count; ++i)
                outputs[i] = OutputLayers[i].Output;
            return outputs;
        }
        private List<LayerBase> InputLayers = new List<LayerBase>();
        private List<LayerBase> OutputLayers = new List<LayerBase>();

        private List<LayerBase> Order = new List<LayerBase>();
        private List<LayerBase> ReversedOrder = new List<LayerBase>();
    }
}
