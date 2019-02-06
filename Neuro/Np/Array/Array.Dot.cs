using System;
using System.Linq;

namespace Neuro
{
	public partial class np
	{
		public partial class Array
		{
			public Array dot(Array nd2)
			{
				var pufferShape = nd2.Storage.Shape;

				// in case must do a reshape
				var oldStorage1 = Storage;
				var oldStorage2 = nd2.Storage;

				if ((NDim == 1) && (nd2.NDim == 1))
				{
					if (Shape[0] != nd2.Shape[0])
						throw new IncorrectShapeException();
					else
					{
						Storage = new Storage();
						Storage.Allocate(new Shape(1, oldStorage1.GetData().Length));
						Storage.SetData(oldStorage1.GetData());

						nd2.Storage = new Storage();
						nd2.Storage.Allocate(new Shape(oldStorage2.GetData().Length));
						nd2.Storage.SetData(oldStorage2.GetData());
					}
				}
				else if (Shape[1] != nd2.Shape[0])
					throw new IncorrectShapeException();

				if ((NDim == 2) && (nd2.NDim == 1))
				{
					var pufferList = pufferShape.Dimensions.ToList();
					pufferList.Add(1);
					nd2.Storage.Reshape(pufferList.ToArray());
				}

				int iterator = Shape[1];
				int dim0 = Shape[0];
				int dim1 = nd2.Shape[1];

				var prod = new Array(new Shape(dim0, dim1));

				float[] nd1Array = Storage.GetData();
				float[] result = prod.Storage.GetData();
				float[] nd2Array = nd2.Storage.GetData();

				for (int idx = 0; idx < prod.Size; idx++)
				{
					int puffer1 = idx % dim0;
					int puffer2 = idx / dim0;
					int puffer3 = puffer2 * iterator;
					for (int kdx = 0; kdx < iterator; kdx++)
						result[idx] += nd2Array[puffer3 + kdx] * nd1Array[dim0 * kdx + puffer1];
				}

				if ((NDim == 1) & (nd2.NDim == 1))
				{
					Storage.Reshape(Storage.GetData().Length);
					nd2.Storage.Reshape(nd2.Storage.GetData().Length);
					prod.Storage.Reshape(1);
				}

				Storage = oldStorage1;
				nd2.Storage = oldStorage2;

				return prod;
			}
		}
	}
}
