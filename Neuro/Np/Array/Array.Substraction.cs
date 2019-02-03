using System;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public static Array operator -(Array a1, Array a2)
            {
                (var scalarNo, var result, var resultArr, var a1Arr, var a2Arr) = GetElementWiseOpData(a1, a2);

                if (scalarNo == 0)
                {
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = a1Arr[idx] - a2Arr[idx];
                }
                else if (scalarNo == 1)
                {
                    float scalar = a1Arr[0];
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = scalar - a2Arr[idx];
                }
                else if (scalarNo == 2)
                {
                    float scalar = a2Arr[0];
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = a1Arr[idx] - scalar;
                }

                return result;
            }
        }
    }
}
