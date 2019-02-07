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
				int[] permutation = new int[NPY_MAXDIMS];

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
					int[] reverse_permutation = new int[NPY_MAXDIMS];

					n = permute.Length;
					var axes = permute;
					if (n != ap.NDim)
					{
						throw new Exception("axes don't match array");
					}
					for (int i = 0; i < n; i++)
					{
						reverse_permutation[i] = -1;
					}
					for (int i = 0; i < n; i++)
					{
						int axis = axes[i];
						if (check_and_adjust_axis(ref axis, ap.NDim) < 0)
						{
							return null;
						}

						if (reverse_permutation[axis] != -1)
							throw new Exception("repeated axis in transpose");

						reverse_permutation[axis] = i;
						permutation[i] = axis;
					}
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
