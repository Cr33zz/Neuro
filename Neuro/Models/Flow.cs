using System;
using System.Collections.Generic;
using System.Linq;
using Neuro.Layers;
using Tensorflow;

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

        public List<Array> Predict(List<Array> inputs)
        {
            return Predicter.Predict(inputs);
        }

        public IEnumerable<LayerBase> GetOutputLayers()
        {
            return OutputLayers;
        }

        public int GetOutputLayersCount()
        {
            return OutputLayers.Count;
        }

        public void Optimize(Optimizers.OptimizerBase optimizer, Loss[] losses)
        {
            Tensor totalLoss = null;

            List<Tensor> train_outputs = new List<Tensor>();
            List<Tensor> targets = new List<Tensor>();

            using (tf.name_scope("loss"))
            {
                for (int i = 0; i < OutputLayers.Count; ++i)
                {
                    var layer = OutputLayers[i];

                    using (tf.name_scope(layer.Name))
                    {
                        targets.Add(tf.placeholder(TF_DataType.TF_FLOAT, new TensorShape(layer.OutputShape.Dims), "target"));
                        var lossTensor = tf.reduce_mean(losses[i].Build(targets[i], layer.Output)); // mean over all batches

                        if (totalLoss == null)
                            totalLoss = lossTensor;
                        else
                            totalLoss = totalLoss + lossTensor;
                    }
                }
            }

            train_outputs.Add(totalLoss);
            Metrics["loss"] = (totalLoss, train_outputs.Count - 1);

            // any additional metrics should go in here

            Trainer = new Trainer(InputLayers.Select(x => x.Input).ToList(), train_outputs, targets, optimizer.GenerateUpdates(Parameters, totalLoss));
            Predicter = new Predicter(InputLayers.Select(x => x.Input).ToList(), OutputLayers.Select(x => x.Output).ToList());
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

            layer.Build();
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

        public List<Tensor> Parameters = new List<Tensor>();

        private List<LayerBase> InputLayers = new List<LayerBase>();
        private List<LayerBase> OutputLayers = new List<LayerBase>();
        public Dictionary<string, (Tensor, int)> Metrics = new Dictionary<string, (Tensor, int)>();

        private List<LayerBase> Order = new List<LayerBase>();
        private List<LayerBase> ReversedOrder;
        public Trainer Trainer;
        public Predicter Predicter;
    }
}
