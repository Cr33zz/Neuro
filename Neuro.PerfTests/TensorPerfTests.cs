using Neuro.Tensors;
using System;
using System.Diagnostics;

namespace Neuro.PerfTests
{
    class TensorPerfTests
    {
        static void Main(string[] args)
        {
            //Tensor.SetOpMode(Tensor.OpMode.MultiCPU);

            Tensor t1 = new Tensor(new Shape(32, 64, 4, 2));
            t1.FillWithRand();
            Tensor t2 = new Tensor(new Shape(64, 32, 4, 2));
            t2.FillWithRand();

            var timer = new Stopwatch();
            timer.Start();

            for (int i = 0; i < 100; ++i)
            {
                t1.Mul(t2);
            }

            timer.Stop();
            Console.WriteLine($"{Math.Round(timer.ElapsedMilliseconds / 1000.0,2)} seconds");

            return;
        }
    }
}
