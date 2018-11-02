using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuro.Tensors;
using System.IO;

namespace Neuro.Tests
{
    [TestClass]
    public class ShapeTests
    {
        [TestMethod]
        public void Reshape_GuessDimension()
        {
            var shape = new Shape(2*5*3*4).Reshaped(new [] {2, -1, 3, 4});
            Assert.AreEqual(5, shape.Height);
        }

        [TestMethod]
        public void NamedDimensions()
        {
            var shape = new Shape(1, 2, 3, 4);
            Assert.AreEqual(1, shape.Width);
            Assert.AreEqual(2, shape.Height);
            Assert.AreEqual(3, shape.Depth);
            Assert.AreEqual(4, shape.BatchSize);
        }

        [TestMethod]
        public void Length()
        {
            var shape = new Shape(2, 3, 4, 5);
            Assert.AreEqual(2*3*4*5, shape.Length);
        }

        [TestMethod]
        public void GetIndex()
        {
            var shape = new Shape(2, 3, 4, 5);
            Assert.AreEqual(shape.GetIndex(1), 1);
            Assert.AreEqual(shape.GetIndex(0, 1), 2);
            Assert.AreEqual(shape.GetIndex(0, 0, 1), 6);
            Assert.AreEqual(shape.GetIndex(0, 0, 0, 1), 24);
            Assert.AreEqual(shape.GetIndex(0, 1, 2, 3), 86);
        }

        [TestMethod]
        public void Dimensions()
        {
            var shape = new Shape(1, 2, 3, 4);
            var dims = shape.Dimensions;
            Assert.AreEqual(1, dims[0]);
            Assert.AreEqual(2, dims[1]);
            Assert.AreEqual(3, dims[2]);
            Assert.AreEqual(4, dims[3]);
        }

        [TestMethod]
        public void Equality()
        {
            var shape1 = new Shape(1, 2, 3, 4);
            var shape2 = new Shape(2, 1, 3, 4);
            var shape3 = new Shape(1, 2, 4, 3);
            var shape4 = new Shape(1, 3, 2, 4);
            var shape5 = new Shape(7, 2, 3, 4);
            var shape6 = new Shape(1, 7, 3, 4);
            var shape7 = new Shape(1, 2, 7, 4);
            var shape8 = new Shape(1, 2, 3, 7);

            Assert.IsFalse(shape1.Equals(shape2));
            Assert.IsFalse(shape1.Equals(shape3));
            Assert.IsFalse(shape1.Equals(shape4));
            Assert.IsFalse(shape1.Equals(shape5));
            Assert.IsFalse(shape1.Equals(shape6));
            Assert.IsFalse(shape1.Equals(shape7));
            Assert.IsFalse(shape1.Equals(shape8));
            Assert.IsTrue(shape1.Equals(shape1));
        }

        [TestMethod]
        public void Serialize_Deserialize()
        {
            string tempFilename = "shape_tmp.txt";

            var shape = new Shape(5, 4, 3, 2);
            using (BinaryWriter writer = new BinaryWriter(File.Open(tempFilename, FileMode.Create)))
            {
                shape.Serialize(writer);
            }

            using (BinaryReader reader = new BinaryReader(File.Open(tempFilename, FileMode.Open)))
            {
                Assert.IsTrue(shape.Equals(Shape.Deserialize(reader)));
            }

            File.Delete(tempFilename);
        }
    }
}
