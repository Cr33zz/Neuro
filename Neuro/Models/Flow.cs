using System;
using System.Collections.Generic;
using System.Linq;
using Neuro.Layers;
using TensorFlow;

namespace Neuro
{
    public class Flow
    {
        public Flow(LayerBase[] inputLayers, LayerBase[] outputLayers)
        {
            InputLayers = inputLayers.ToList();
            OutputLayers = outputLayers.ToList();

            List<LayerBase> visited = new List<LayerBase>();

            foreach (var inputLayer in InputLayers)
                ProcessLayer(inputLayer, ref visited);

            ReversedOrder = new List<LayerBase>(Order);
            ReversedOrder.Reverse();
        }

        // For cloning purposes
        private Flow()
        {
        }

        public List<Array> FeedForward(List<Array> inputs)
        {
            throw new NotImplementedException();
        }

        public IEnumerable<LayerBase> GetOutputLayers()
        {
            return OutputLayers;
        }

        public int GetOutputLayersCount()
        {
            return OutputLayers.Count;
        }

        public void Optimize()
        {
            throw new NotImplementedException();
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
            Parameters.AddRange(layer.TrainableParams);

            visited.Add(layer);

            foreach (var outLayer in layer.OutputLayers)
                ProcessLayer(outLayer, ref visited);
        }

        public Flow Clone()
        {
			// clone is not a frequently used functionality so I'm not too concerned about its performance

			// make clones first and store then in dictionary
			//var clones = new Dictionary<string, LayerBase>();
			//foreach (var layer in Order)
			//{
			//    var clone = layer.Clone();
			//    clones[clone.Name] = clone;
			//}

			//// then connect them in the same manner as in original network and clone order
			//var flowClone = new Flow();

			//foreach (var layer in Order)
			//{
			//    var layerClone = clones[layer.Name];
			//    foreach (var inLayer in layer.InputLayers)
			//    {
			//        var inLayerClone = clones[inLayer.Name];
			//        layerClone.InputLayers.Add(inLayerClone);
			//        inLayerClone.OutputLayers.Add(layerClone);
			//    }

			//    flowClone.Order.Add(layerClone);
			//}

			//flowClone.ReversedOrder = new List<LayerBase>(flowClone.Order);
			//flowClone.ReversedOrder.Reverse();

			//foreach (var layer in InputLayers)
			//{
			//    var layerClone = clones[layer.Name];
			//    flowClone.InputLayers.Add(layerClone);
			//}

			//foreach (var layer in OutputLayers)
			//{
			//    var layerClone = clones[layer.Name];
			//    flowClone.OutputLayers.Add(layerClone);
			//}

			//return flowClone;
			throw new NotImplementedException();
		}

		public IEnumerable<LayerBase> GetLayers()
        {
			return Order;
		}

		public string Summary()
        {
            int totalParams = 0;
            string output = "____________________________________________________________________________________________________\n";
            output += "Layer                        Output TFShape              Param #     Connected to\n";
            output += "====================================================================================================\n";

            foreach (var layer in Order)
            {
                totalParams += layer.GetParamsNum();
                output += $"{(layer.Name + " (" + layer.GetType().Name + ")").PadRight(29)}" + $"({layer.OutputShape.Dims.Get(-1)}, {layer.OutputShape.Dims.Get(-2)}, {layer.OutputShape.Dims.Get(-3)})".PadRight(26) + $"{layer.GetParamsNum()}".PadRight(13) + (layer.InputLayers.Count > 0 ? layer.InputLayers[0].Name : "") + "\n";
                for (int i = 1; i < layer.InputLayers.Count; ++i)
                    output += layer.InputLayers[i].Name.PadLeft(68 + layer.InputLayers[i].Name.Length) + "\n";
                output += "____________________________________________________________________________________________________\n";
            }

            output += $"Total params: {totalParams}";

            return output;
        }

        //public override TFTensor[] GetOutputs()
        //{
        //    TFTensor[] outputs = new TFTensor[OutputLayers.Count];
        //    for (int i = 0; i < OutputLayers.Count; ++i)
        //        outputs[i] = OutputLayers[i].Output;
        //    return outputs;
        //}

        public List<Tensor> Parameters = new List<Tensor>();

        private List<LayerBase> InputLayers = new List<LayerBase>();
        private List<LayerBase> OutputLayers = new List<LayerBase>();

        private List<LayerBase> Order = new List<LayerBase>();
        private List<LayerBase> ReversedOrder;
    }
}
