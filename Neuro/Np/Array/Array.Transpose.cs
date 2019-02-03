using System;
using System.Linq;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public Array T()
            {
                var result = new Array();

                if (NDim == 1)
                {
                    result.Storage.Reshape(1, result.Shape[0]);
                }
                else
                {
                    result.Storage.Reshape(result.Shape.Reverse().ToArray());
                    for (int idx = 0; idx < result.Shape[0]; idx++)
                    for (int jdx = 0; jdx < result.Shape[1]; jdx++)
                        result[idx, jdx] = this[jdx, idx];
                }

                return result;
            }
        }
    }
}
