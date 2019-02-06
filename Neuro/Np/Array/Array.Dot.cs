using System;
using System.Collections.Generic;
using System.Linq;

//based on PyArray_MatrixProduct2 from https://raw.githubusercontent.com/numpy/numpy/master/numpy/core/src/multiarray/multiarraymodule.c
namespace Neuro
{
	public partial class np
	{
		public partial class Array
		{
			public Array dot(Array a2)
			{
				Array ap1 = this;
				Array ap2 = a2;
				float[] ap1Buf = ap1.Data();
				float[] ap2Buf = ap2.Data();

				int nd, axis, matchDim;
				int is1, is2;
				int[] dimensions = new int[NPY_MAXDIMS];
				
				if (ap1.NDim == 0 || a2.NDim == 0)
				{
					return ap1 * ap2;
				}
				int l = ap1.Dims[NDim - 1];
				if (a2.NDim > 1)
				{
					matchDim = a2.NDim - 2;
				}
				else
				{
					matchDim = 0;
				}
				if (ap2.Dims[matchDim] != l)
				{
					throw new Exception("dot: dimensions alignment error");
				}
				nd = NDim + a2.NDim - 2;
				if (nd > NPY_MAXDIMS)
				{
					throw new Exception("dot: too many dimensions in result");
				}
				int j = 0;
				for (int i = 0; i < NDim - 1; i++)
				{
					dimensions[j++] = ap1.Dims[i];
				}
				for (int i = 0; i < a2.NDim - 2; i++)
				{
					dimensions[j++] = ap2.Dims[i];
				}
				if (a2.NDim > 1)
				{
					dimensions[j++] = ap2.Dims[a2.NDim - 1];
				}

				is1 = ap1.Strides[NDim - 1];
				is2 = ap2.Strides[matchDim];
				/* Choose which subtype to return */
				int[] outputDims = new int[nd];
				for (int i = 0; i < nd; ++i)
					outputDims[i] = dimensions[i];

				Array result = new Array(outputDims);
				float[] outbuf = result.Data();				
				int op = 0;
				axis = NDim - 1;
				Iter it1 = ap1.IterAllButAxis(axis);
				Iter it2 = ap2.IterAllButAxis(matchDim);

				while (it1.index < it1.size)
				{
					while (it2.index < it2.size)
					{
						FloatDot(ap1Buf,it1.dataptr, is1, ap2Buf, it2.dataptr, is2, outbuf, op, l);
						op += 1;
						it2.Next();
					}
					it1.Next();
					it2.Reset();
				}

				return result;
			}

			private delegate void FloatDotFunc(float[] a, int idxa, int stridea, float[] b, int idxb, int strideb, float[] res, int idxres, int n);
			private static FloatDotFunc FloatDot = SimpleFloatDot;

			private static void SimpleFloatDot(float[] a, int idxa, int stridea, float[] b, int idxb, int strideb, float[] res, int idxres, int n)
			{
				float prod = 0;
				for (int i = 0; i < n; ++i)
					prod += a[idxa + stridea * i] * b[idxb + strideb * i];
				res[idxres] = prod;
			}

			//https://www.csie.ntu.edu.tw/~azarc/sna/numpy-1.3.0/numpy/core/blasdot/_dotblas.c
			private static void BlasFloatDot(float[] a, int idxa, int stridea, float[] b, int idxb, int strideb, float[] res, int idxres, int n)
			{
				//int na = stridea;
				//int nb = strideb;

				//if ((na >= 0) && (nb >= 0))
				//	cblas_sdot((int)n, (float*)a, na, (float*)b, nb);
				//else
				//	SimpleFloatDot(a, idxa, stridea, b, idxb, strideb, res, idxres, n);
				throw new NotImplementedException();
			}
		}
	}
}
