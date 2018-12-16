using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neuro.Tensors;

namespace Neuro.Layers
{
    public class Merge : LayerBase
    {
        public enum Mode
        {
            Sum,
            Avg,
            Max,
            Min
        }

        public Merge(LayerBase[] inputLayers, Mode mergeMode)
            : base(inputLayers, inputLayers[0].OutputShape)
        {
            MergeMode = mergeMode;
        }

        public Merge(Shape[] inputShapes, Mode mergeMode)
            : base(inputShapes, inputShapes[0])
        {
            MergeMode = mergeMode;
        }

        public override LayerBase Clone()
        {
            return new Merge(InputShapes, MergeMode);
        }

        protected override void FeedForwardInternal()
        {   
            
        }

        protected override void BackPropInternal(Tensor outputGradient)
        {            
            
        }

        public readonly Mode MergeMode;
    }
}
