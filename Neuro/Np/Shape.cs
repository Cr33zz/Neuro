using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuro
{
    public partial class np
    {
        public class Shape
        {
            public Shape(params int[] dims)
            {
                ReShape(dims);
            }

            public Shape(IEnumerable<int> shape)
                : this(shape.ToArray())
            {
            }

            public int[] Dimensions { get; private set; }
            public int[] Strides { get; private set; }
            public int Size { get; private set; }
            public int NDim => Dimensions.Length;

            public (int, int) BiShape => Dimensions.Length == 2 ? (Dimensions[0], Dimensions[1]) : (0, 0);
            public (int, int, int) TriShape => Dimensions.Length == 3 ? (Dimensions[0], Dimensions[1], Dimensions[2]) : (0, 0, 0);

            public static implicit operator int(Shape shape) => shape.Size;
            public static implicit operator (int, int) (Shape shape) => shape.Dimensions.Length == 2 ? (shape.Dimensions[0], shape.Dimensions[1]) : (0, 0);
            public static implicit operator (int, int, int) (Shape shape) => shape.Dimensions.Length == 3 ? (shape.Dimensions[0], shape.Dimensions[1], shape.Dimensions[2]) : (0, 0, 0);
            public static implicit operator int[] (Shape shape) => shape.Dimensions;
            public static implicit operator Shape(int[] dims) => new Shape(dims);
            public static implicit operator Shape(int dim) => new Shape(dim);

            public int GetIndexInShape(params int[] select)
            {
				if (NDim == 0)
					return 0;

				int idx = 0;

                for (int i = 0; i < select.Length; i++)
                {
                    idx += Strides[i] * (select[i] < 0 ? Dimensions[i] + select[i] : select[i]);
                }

                return idx;
            }

            public int[] GetDimIndexOutShape(int select)
            {
                int[] dimIndexes = null;
                if (Strides.Length == 1)
                {
	                dimIndexes = new[] {select};
                }
                else
                {
                    int counter = select;
                    dimIndexes = new int[Strides.Length];

                    for (int idx = 0; idx < Strides.Length; idx++)
                    {
                        dimIndexes[idx] = counter / Strides[idx];
                        counter -= dimIndexes[idx] * Strides[idx];
                    }
                }

                return dimIndexes;
            }

            public void ReShape(params int[] dims)
            {
                Dimensions = dims;
				Size = 1;

                foreach (int dimSize in dims)
                    Size *= dimSize;

				Strides = new int[Dimensions.Length];
				if (Dimensions.Length > 0)
				{
					Strides[Strides.Length - 1] = 1;
					for (int idx = Strides.Length - 1; idx >= 1; idx--)
						Strides[idx - 1] = Strides[idx] * Dimensions[idx];
				}
			}

            public static bool operator ==(Shape a, Shape b)
            {
                if (b is null)
                    return false;
                return a.Dimensions.SequenceEqual(b.Dimensions);
            }

            public static bool operator !=(Shape a, Shape b)
            {
                return !(a == b);
            }

            public override bool Equals(object obj)
            {
                if (obj is Shape s)
                    return this == s;
                return false;
            }

            public override int GetHashCode()
            {
                return base.GetHashCode();
            }

            public override string ToString()
            {
                return "(" + String.Join(", ", Dimensions) + ")";
            }
        }
    }
}
