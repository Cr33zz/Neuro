//based on https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/iterators.c and 
//https://github.com/numpy/numpy/blob/master/numpy/core/include/numpy/ndarraytypes.h

using System;

namespace Neuro
{
	public partial class np
	{
		public partial class Array
		{
			public class MultiIter
			{
				public MultiIter(params Array[] arrays)
				{
					int n = arrays.Length;

					if (n < 1 || n > NPY_MAXARGS)
						throw new Exception($"Need at least 1 and at most {NPY_MAXARGS} array objects.");

					numiter = n;
					index = 0;
					iters = new Iter[n];

					for (int i = 0; i < n; i++)
					{
						if (arrays[i] == null)
							throw new Exception("Null array on the list of arrays.");

						iters[i] = new Iter(arrays[i]);
					}

					Broadcast();
				}

				public int numiter;
				public int size;
				public int index;
				public int nd;
				public int[] dimensions;
				public Iter[] iters;

				public void Next()
				{
					index++;
					for (int __npy_mi = 0; __npy_mi < numiter; __npy_mi++)
					{
						iters[__npy_mi].Next();
					}
				}

				public void NextI(int i)
				{
					iters[i].Next();
				}

				public void GoTo(params int[] dest)
				{
					for (int __npy_mi = 0; __npy_mi < numiter; __npy_mi++)
					{
						iters[__npy_mi].GoTo(dest);
					}
					index = iters[0].index;
				}

				public void GoTo1D(int ind)
				{
					for (int __npy_mi = 0; __npy_mi < numiter; __npy_mi++)
					{
						iters[__npy_mi].GoTo1D(ind);
					}
					index = iters[0].index;
				}

				public float Data(int i)
				{
					return iters[i].Data();
				}

				public bool NotDone()
				{
					return index < size;
				}

				public void Reset()
				{
					index = 0;
					for (int __npy_mi = 0; __npy_mi < numiter; __npy_mi++)
					{
						iters[__npy_mi].Reset();
					}
				}

				private void Broadcast()
				{
					nd = 0;

					/* Discover the broadcast number of dimensions */
					for (int i = 0; i < numiter; i++)
					{
						nd = Math.Max(nd, iters[i].ao.NDim);
					}

					dimensions = new int[nd];

					/* Discover the broadcast shape in each dimension */
					for (int i = 0; i < nd; i++)
					{
						dimensions[i] = 1;
						for (int j = 0; j < numiter; j++)
						{
							var it = iters[j];
							/* This prepends 1 to shapes not already equal to nd */
							int k = i + it.ao.NDim - nd;
							if (k >= 0)
							{
								int t = it.ao.Dims[k];
								if (t == 1)
								{
									continue;
								}
								if (dimensions[i] == 1)
								{
									dimensions[i] = t;
								}
								else if (dimensions[i] != t)
								{
									throw new Exception($"shape mismatch: objects cannot be broadcast to a single shape");
								}
							}
						}
					}

					/*
					 * Reset the iterator dimensions and strides of each iterator
					 * object -- using 0 valued strides for broadcasting
					 */
					size = MultiplyList(dimensions);
					for (int i = 0; i < numiter; i++)
					{
						var it = iters[i];
						it.nd_m1 = nd - 1;
						it.size = size;
						if (it.ao.NDim != 0)
							it.factors[nd - 1] = 1;

						for (int j = 0; j < nd; j++)
						{
							it.dims_m1[j] = dimensions[j] - 1;
							int k = j + it.ao.NDim - nd;

							/*
							 * If this dimension was added or shape of
							 * underlying array was 1
							 */
							if ((k < 0) || it.ao.Dims[k] != dimensions[j])
							{
								it.contiguous = false;
								it.strides[j] = 0;
							}
							else
							{
								it.strides[j] = it.ao.Strides[k];
							}

							it.backstrides[j] = it.strides[j] * it.dims_m1[j];

							if (j > 0)
								it.factors[nd - j - 1] = it.factors[nd - j] * dimensions[nd - j];
						}
						it.Reset();
					}
				}
			}
		}
	}
}
