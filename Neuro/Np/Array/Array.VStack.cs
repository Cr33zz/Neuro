using System;
using System.Collections.Generic;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public static Array VStack(params Array[] arrays)
            {
                if (arrays == null || arrays.Length == 0)
                    throw new Exception("Input arrays can not be empty");

                var list = new List<float>();
                var result = new Array();
                foreach (Array ele in arrays)
                {
                    if (arrays[0].Dims != ele.Dims)
                        throw new Exception("Arrays mush have same shapes");
                    list.AddRange(ele.Storage.GetData());
                }
                result.Storage.SetData(list.ToArray());
                if (arrays[0].NDim == 1)
                {
                    result.Storage.Reshape(arrays.Length, arrays[0].Dims[0]);
                }
                else
                {
                    int[] shapes = arrays[0].Dims;
                    shapes[0] *= arrays.Length;
                    result.Storage.Reshape(shapes);
                }
                return result;
            }
        }
    }
}
