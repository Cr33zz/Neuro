using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

//based on https://raw.githubusercontent.com/numpy/numpy/fd89a4137969b676d4449e2b61ecd7f4c5811d7a/numpy/core/src/multiarray/common.h
namespace Neuro
{
	public partial class np
	{
		public partial class Array
		{
			private bool Likely(bool cnd)
			{
				return cnd == true;
			}

			private bool Unlikely(bool cnd)
			{
				return cnd == false;
			}

			/*
			 * Returns -1 and sets an exception if *index is an invalid index for
			 * an array of size max_item, otherwise adjusts it in place to be
			 * 0 <= *index < max_item, and returns 0.
			 * 'axis' should be the array axis that is being indexed over, if known. If
			 * unknown, use -1.
			 * If _save is NULL it is assumed the GIL is taken
			 * If _save is not NULL it is assumed the GIL is not taken and it
			 * is acquired in the case of an error
			 */
			private int check_and_adjust_index(ref int index, int max_item, int axis)
			{
				/* Check that index is valid, taking into account negative indices */
				if (Unlikely((index < -max_item) || (index >= max_item)))
				{
					/* Try to be as clear as possible about what went wrong. */
					if (axis >= 0)
					{
						throw new Exception($"index {index} is out of bounds for axis {axis} with size {max_item}");
					}

					throw new Exception($"index {index} is out of bounds for size {max_item}");
				}

				/* adjust negative indices */
				if (index < 0)
				{
					index += max_item;
				}

				return 0;
			}

			/*
			 * Returns -1 and sets an exception if *axis is an invalid axis for
			 * an array of dimension ndim, otherwise adjusts it in place to be
			 * 0 <= *axis < ndim, and returns 0.
			 *
			 * msg_prefix: borrowed reference, a string to prepend to the message
			 */
			private int check_and_adjust_axis(ref int axis, int ndim)
			{
				/* Check that index is valid, taking into account negative indices */
				if (Unlikely((axis < -ndim) || (axis >= ndim)))
				{
					throw new Exception($"axis {axis} is out of bounds for size {ndim}");
				}

				/* adjust negative indices */
				if (axis < 0)
				{
					axis += ndim;
				}

				return 0;
			}
		}
	}
}
