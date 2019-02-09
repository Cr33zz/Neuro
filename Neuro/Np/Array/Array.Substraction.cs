using System;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
            public static Array operator -(Array a1, Array a2)
            {
				MultiIter mit = new MultiIter(a1, a2);
				Array result = new Array(mit.dimensions);
				float[] resultArr = result.Data();

				while (mit.NotDone())
				{
					resultArr[mit.index] = mit.Data(0) - mit.Data(1);
					mit.Next();
				}

				return result;
			}
        }
    }
}
