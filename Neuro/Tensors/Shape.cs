using System;
using System.Diagnostics;
using System.IO;

namespace Neuro.Tensors
{
    public class Shape : IEquatable<Shape>
    {
        public static int Auto = -1; // Automatically guesses

        public Shape(int width, int height = 1, int depth = 1, int batchSize = 1)
        {
            Dimensions = new[] { width, height, depth, batchSize };
            Dim0 = width;
            Dim0Dim1 = Dim0 * height;
            Dim0Dim1Dim2 = Dim0Dim1 * depth;
            Length = Dim0Dim1Dim2 * batchSize;
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

        public int GetIndex(int w, int h = 0, int d = 0, int n = 0)
        {
            Debug.Assert(w < Width);
            Debug.Assert(h < Height);
            Debug.Assert(d < Depth);
            Debug.Assert(n < BatchSize);
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

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Width);
            writer.Write(Height);
            writer.Write(Depth);
            writer.Write(BatchSize);
        }

        public static Shape Deserialize(BinaryReader reader)
        {
            int[] dims = new [] { reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32() };
            return Shape.From(dims);
        }

        public int Width => Dimensions[0];

        public int Height => Dimensions[1];

        public int Depth => Dimensions[2];

        public int BatchSize => Dimensions[3];

        public int[] Dimensions { get; }

        public int Length { get; }

        public override string ToString() { return $"{Width}x{Height}x{Depth}x{BatchSize}"; }

        private readonly int Dim0;
        private readonly int Dim0Dim1;
        private readonly int Dim0Dim1Dim2;
    }
}
