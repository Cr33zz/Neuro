using System;

namespace Neuro
{
    public partial class np
    {
        public partial class Array
        {
	        public Array T => Transpose();

			public Array Transpose(params int[] permute)
            {
	            Array ap = this;
	            int n;
				int[] permutation = new int[NPY_MAXDIMS], reverse_permutation = new int[NPY_MAXDIMS];

				if (permute.Length == 0)
				{
					n = ap.NDim;
					for (int i = 0; i < n; i++)
					{
						permutation[i] = n - 1 - i;
					}
				}
				else
				{
					throw new NotImplementedException();
					//int axes;
					//n = permute.Length;
					//axes = permute->ptr;
					//if (n != ap.NDim)
					//{
					//	throw new Exception("axes don't match array");
					//}
					//for (int i = 0; i < n; i++)
					//{
					//	reverse_permutation[i] = -1;
					//}
					//for (int i = 0; i < n; i++)
					//{
					//	int axis = axes[i];
					//	if (check_and_adjust_axis(&axis, PyArray_NDIM(ap)) < 0)
					//	{
					//		return NULL;
					//	}
					//	if (reverse_permutation[axis] != -1)
					//	{
					//		PyErr_SetString(PyExc_ValueError,
					//						"repeated axis in transpose");
					//		return NULL;
					//	}
					//	reverse_permutation[axis] = i;
					//	permutation[i] = axis;
					//}
				}

				var ret = (Array)ap.Clone();
				
				/* fix the dimensions and strides of the return-array */
				for (int i = 0; i < n; i++)
				{
					ret.Dims[i] = ap.Dims[permutation[i]];
					ret.Strides[i] = ap.Strides[permutation[i]];
				}
				
				return ret;
			}
        }
    }
}
