using Neuro.Tensors;
using System;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class TensorPerfTests
    {
        static void Main(string[] args)
        {
            Tensor.SetOpMode(Tensor.OpMode.MultiCPU);

            for (int i = 1; i < 10; ++i)
            {
                int batchSize = i * 1;
                Tensor t1 = new Tensor(new Shape(32, 32, 1, batchSize));
                t1.FillWithRand();
                Tensor t2 = new Tensor(new Shape(32, 32, 1, batchSize));
                t2.FillWithRand();

                Tensor res = new Tensor(t2.Shape);

                var timer = new Stopwatch();
                timer.Start();

                for (int n = 0; n < 20; ++n)
                {
                    t1.Mul(t2, res);
                }

                timer.Stop();
                Trace.WriteLine($"Elements: {t1.Shape.Length} {Math.Round(timer.ElapsedMilliseconds / 1000.0, 2)} seconds");
            }

            return;
        }
    }
}
