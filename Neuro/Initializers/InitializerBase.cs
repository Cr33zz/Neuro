using System;
using System.Linq;

namespace Neuro
{
    public abstract class InitializerBase
    {
        public abstract Tensor Init(int[] shape);

        protected static (float fanIn, float fanOut) ComputeFans(int[] shape)
        {
            float fanIn, fanOut;
            if (shape.Length == 2)
            {
                fanIn = shape[0];
                fanOut = shape[1];
            }
            else if (new[] { 3, 4, 5 }.Contains(shape.Length))
            {
                int receptiveFieldSize = shape.Get(0, 2).Product();
                fanIn = shape.Get(-2) * receptiveFieldSize;
                fanOut = shape.Get(-1) * receptiveFieldSize;
            }
            else
            {
                // No specific assumptions.
                fanIn = (float)Math.Sqrt(shape.Product());
                fanOut = (float)Math.Sqrt(shape.Product());
            }

            return (fanIn, fanOut);
        }
    }
}
