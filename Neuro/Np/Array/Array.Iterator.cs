using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

//based on https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/iterators.c and 
//https://github.com/numpy/numpy/blob/master/numpy/core/include/numpy/ndarraytypes.h
namespace Neuro
{
	public partial class np
	{
		public partial class Array
		{
			private const int NPY_MAXDIMS = 16; // 32 in NumPy

			public Iter IterNew()
			{
				return new Iter(this);
			}

			// Get Iterator that iterates over all but one axis (don't use this with GoTo1D).
			// The axis will be over-written if negative with the axis having the smallest stride.
			public Iter IterAllButAxis(int axis)
			{
				Array arr = this;
				var it = IterNew();

				if (arr.NDim == 0)
				{
					return it;
				}
				if (axis < 0)
				{
					int minaxis = 0;
					int minstride = 0;
					int i = 0;
					while (minstride == 0 && i < arr.NDim)
					{
						minstride = arr.Strides[i];
						i++;
					}
					for (i = 1; i < arr.NDim; i++)
					{
						if (arr.Strides[i] > 0 &&
						    arr.Strides[i] < minstride)
						{
							minaxis = i;
							minstride = arr.Strides[i];
						}
					}
					axis = minaxis;
				}

				/* adjust so that will not iterate over axis */
				it.contiguous = false;
				if (it.size != 0)
				{
					it.size /= arr.Dims[axis];
				}
				it.dims_m1[axis] = 0;
				it.backstrides[axis] = 0;

				// (won't fix factors so don't use GoTo1D with this iterator)
				return it;
			}

			public class Iter
			{
				public Iter(Array a)
				{
					ao = a;
					int nd = a.NDim;
					contiguous = true;
					size = a.Size;
					nd_m1 = nd - 1;
					if (nd != 0)
					{
						factors[nd - 1] = 1;
					}
					for (int i = 0; i < nd; i++)
					{
						dims_m1[i] = ao.Dims[i] - 1;
						strides[i] = ao.Strides[i];
						backstrides[i] = strides[i] * dims_m1[i];
						if (i > 0)
						{
							factors[nd - i - 1] = factors[nd - i] * ao.Dims[nd - i];
						}
						bounds[i,0] = 0;
						bounds[i,1] = ao.Dims[i] - 1;
						limits[i,0] = 0;
						limits[i,1] = ao.Dims[i] - 1;
						limits_sizes[i] = limits[i,1] - limits[i,0] + 1;
					}

					translate = SimpleTranslate;
					Reset();
				}

				public Array ao;
				public int nd_m1;
				public int index;
				public int size;
				public int[] coordinates = new int[NPY_MAXDIMS];
				public int[] dims_m1 = new int[NPY_MAXDIMS];
				public int[] strides = new int[NPY_MAXDIMS];
				public int[] backstrides = new int[NPY_MAXDIMS];
				public int[] factors = new int[NPY_MAXDIMS];
				public bool contiguous;
				public int dataptr; // this is index of current item in the under laying flat array
				public int[,] bounds = new int[NPY_MAXDIMS,2];
				public int[,] limits = new int[NPY_MAXDIMS,2];
				public int[] limits_sizes = new int[NPY_MAXDIMS];
				public TranslateFunc translate;

				public delegate float TranslateFunc(Iter iter, int[] coordinates);

				public void Next1()
				{
					dataptr += strides[0];
					coordinates[0]++;
				}

				public void Next2()
				{
					if (coordinates[1] < dims_m1[1])
					{
						coordinates[1]++;
						dataptr += strides[1];
					}
					else
					{
						coordinates[1] = 0;
						coordinates[0]++;
						dataptr += strides[0] - backstrides[1];
					}
				}

				public void Next()
				{
					index++;
					if (nd_m1 == 0)
					{
						Next1();
					}
					else if (contiguous)
					{
						dataptr += 1;
					}
					else if (nd_m1 == 1)
					{
						Next2();
					}
					else
					{
						for (int __npy_i = nd_m1; __npy_i >= 0; __npy_i--)
						{
							if (coordinates[__npy_i] < dims_m1[__npy_i])
							{
								coordinates[__npy_i]++;
								dataptr += strides[__npy_i];
								break;
							}
							else
							{
								coordinates[__npy_i] = 0;
								dataptr -= backstrides[__npy_i];
							}
						}
					}
				}

				public void GoTo(params int[] destination)
				{
					index = 0;
					dataptr = 0;
					for (int __npy_i = nd_m1; __npy_i >= 0; __npy_i--)
					{
						if (destination[__npy_i] < 0)
						{
							destination[__npy_i] += dims_m1[__npy_i] + 1;
						}
						dataptr += destination[__npy_i] * strides[__npy_i];
						coordinates[__npy_i] = destination[__npy_i];
						index += destination[__npy_i] * (__npy_i == nd_m1 ? 1 : dims_m1[__npy_i + 1] + 1);
					}
				}

				public void GoTo1D(int ind)
				{
					int __npy_ind = ind;

					if (__npy_ind < 0)
						__npy_ind += size;
					index = __npy_ind;
					if (nd_m1 == 0)
					{
						dataptr = __npy_ind * strides[0];
					}
					else if (contiguous)
					{
						dataptr = __npy_ind * 1;
					}
					else
					{
						dataptr = 0;
						for (int __npy_i = 0; __npy_i <= nd_m1; __npy_i++)
						{
							dataptr += (__npy_ind / factors[__npy_i]) * strides[__npy_i];
							__npy_ind %= factors[__npy_i];
						}
					}
				}

				public float Data()
				{
					return ao.Data()[dataptr];
				}

				public bool NotDone()
				{
					return index < size;
				}

				public void Reset()
				{
					index = 0;
					dataptr = 0;
					for (int i = 0; i < nd_m1 + 1; ++i)
						coordinates[i] = 0;
				}

				private static float SimpleTranslate(Iter iter, int[] coordinates)
				{
					int idx = 0;

					for (int i = 0; i < iter.ao.NDim; ++i)
					{
						idx += coordinates[i] * iter.strides[i];
					}

					return iter.ao.Data()[idx];
				}
			}
		}
	}
}
