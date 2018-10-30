using System;

namespace Neuro.Tensors
{
    public class Shape : IEquatable<Shape>
    {
        public static int Auto = -1; // Automatically guesses

        public Shape(int width, int height = 1, int depth = 1, int batches = 1)
        {
            Dimensions = new[] { width, height, depth, batches };
            Dim0 = width;
            Dim0Dim1 = Dim0 * height;
            Dim0Dim1Dim2 = Dim0Dim1 * depth;
            Length = Dim0Dim1Dim2 * batches;
        }

        public static Shape From(int[] dimensions)
        {
            switch (dimensions.Length)
            {
                case 1: return new Shape(dimensions[0]);
                case 2: return new Shape(dimensions[0], dimensions[1]);
                case 3: return new Shape(dimensions[0], dimensions[1], dimensions[2]);
                case 4: return new Shape(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
            }

            throw new ArgumentException($"Invalid number of dimensions {dimensions.Length}.");
        }

        public Shape Reshaped(int[] dimensions)
        {
            int dToUpdate = -1;
            int product = 1;
            for (int d = 0; d < 4; ++d)
            {
                if (dimensions[d] == -1)
                {
                    dToUpdate = d;
                    continue;
                }

                product *= dimensions[d];
            }

            if (dToUpdate >= 0)
            {
                dimensions[dToUpdate] = Length / product;
            }

            return From(dimensions);
        }

        public int GetIndex(int w, int h = 1, int d = 1, int n = 1)
        {
            return Dim0Dim1Dim2 * n + Dim0Dim1 * d + Dim0 * h + w;
        }

        public bool Equals(Shape other)
        {
            if (ReferenceEquals(null, other) || this.Length != other.Length)
                return false;

            if (ReferenceEquals(this, other))
                return true;
            
            return Width == other.Width && Height == other.Height && Depth == other.Depth;
        }

        public int Width => Dimensions[0];

        public int Height => Dimensions[1];

        public int Depth => Dimensions[2];

        public int Batches => Dimensions[3];

        public int[] Dimensions { get; }

        public int Length { get; }

        private readonly int Dim0;
        private readonly int Dim0Dim1;
        private readonly int Dim0Dim1Dim2;
    }
}
