using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI
{
    public interface Network
    {
        public float LearningRate { get; set; }
        public float Beta { get; set; }

        public float[] FeedForward(float[] input);

        public void Train(float[][] inputs, float[][] targets);

        public void TrainLoss(float[][] inputs, Func<int, float[], float[]> lossFunction);

        public void CheckNetwork();

        public void Save(string outputDir);

        public void Load(string inputDir);

        public Network Copy();
    }

    public static class NetworkHelp
    {
        public static float Sigmoid(float x)
            => (float)(1 / (1 + Math.Exp(-x)));

        private static float SigmoidPrime(float x)
            => Sigmoid(x) * (1 - Sigmoid(x));

        private static float ReLU(float x)
        {
            if (x >= 0)
                return x;
            return 0;
        }

        private static float ReLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return 0;
        }

        private static float eLU(float x)
        {
            if (x >= 0)
                return x;
            return (float)Math.Exp(x) - 1;
        }

        private static float eLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return (float)Math.Exp(x);
        }

        private static float Linear(float x)
            => x;

        private static float LinearPrime(float x)
            => 1;

        public static Func<float, float> Derivatives(Func<float, float> function)
        {
            if (function == Sigmoid)
                return SigmoidPrime;
            if (function == ReLU)
                return ReLUPrime;
            if (function == eLU)
                return eLUPrime;
            if (function == Linear)
                return LinearPrime;

            throw new Exception("Could not find derivative of Activation Function");
        }
    }
}
