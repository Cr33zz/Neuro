using System;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            private static (int, Array, float[], float[], float[]) GetElementWiseOpData(Array a1, Array a2)
            {
                int scalarNo = a1.NDim == 0 ? 1 : (a2.NDim == 0 ? 2 : 0);

                if (scalarNo == 0 && a1.Shape != a2.Shape)
                    throw new IncorrectShapeException();

                Array result = scalarNo == 1 ? new Array(a2.Shape) : new Array(a1.Shape);

                return (scalarNo, result, result.Storage.GetData(), a1.Storage.GetData(), a2.Storage.GetData());
            }

            public static Array operator +(Array a1, Array a2)
            {
                (var scalarNo, var result, var resultArr, var a1Arr, var a2Arr) = GetElementWiseOpData(a1, a2);

                if (scalarNo == 0)
                {
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = a1Arr[idx] + a2Arr[idx];
                }
                else if (scalarNo == 1)
                {
                    float scalar = a1Arr[0];
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = scalar + a2Arr[idx];
                }
                else if (scalarNo == 2)
                {
                    float scalar = a2Arr[0];
                    for (int idx = 0; idx < resultArr.Length; idx++)
                        resultArr[idx] = a1Arr[idx] + scalar;
                }

                return result;
            }
        }
    }
}
