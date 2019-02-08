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

                if (scalarNo == 0 && a1.Dims != a2.Dims)
                    throw new IncorrectShapeException();

                Array result = scalarNo == 1 ? new Array(a2.Dims) : new Array(a1.Dims);

                return (scalarNo, result, result.Storage.GetData(), a1.Storage.GetData(), a2.Storage.GetData());
            }

            public static Array operator +(Array a1, Array a2)
            {
				MultiIter mit = new MultiIter(a1, a2);
				Array result = new Array(mit.dimensions);
				float[] resultArr = result.Data();

				while (mit.NotDone())
				{
					resultArr[mit.index] = mit.Data(0) + mit.Data(1);
					mit.Next();
				}

                return result;
            }
        }
    }
}
