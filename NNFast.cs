using System.Numerics.Tensors;

namespace AI
{
    public class NNFast
    {
        public float LearningRate;
        public int[] Layers;

        public float[][] Neurons;
        public float[][] Z;
        public float[][] Biases;
        public float[][] Weights;

        public Action<float[], float[]>[] ActivationFunctions;
        public Action<float[], float[]>[] ActivationFunctionsDerivatives;

        public NNFast(int[] layers, ActivationTypes[] activationFunctionsTypes, float learningRate)
        {
            if (layers == null)
                throw new Exception("Layers given are null");

            if (activationFunctionsTypes == null)
                throw new Exception("Activation Functions array is null");

            if (layers.Length - 1 != activationFunctionsTypes.Length)
                throw new Exception("Number of layers and activation functions don't match");

            Layers = layers;
            LearningRate = learningRate;

            Neurons = new float[Layers.Length][];
            Weights = new float[Layers.Length][];
            Biases = new float[Layers.Length][];
            Z = new float[Layers.Length][];

            for (int l = 0; l < Layers.Length; l++)
            {
                Neurons[l] = new float[Layers[l]];

                if (l != 0)
                {
                    Z[l] = new float[Layers[l]];
                    Biases[l] = new float[Layers[l]]; //init biases at 0
                    Weights[l] = new float[Layers[l] * Layers[l - 1]];

                    float std = MathF.Sqrt(2.0f / Layers[l - 1]); // He initialization
                    for (int i = 0; i < Layers[l]; i++)
                    {
                        for (int j = 0; j < Layers[l - 1]; j++)
                            Weights[l][i * Layers[l - 1] + j] = Utils.GaussianRandom(0, std);
                    }

                }
            }

            ActivationFunctions = new Action<float[], float[]>[Layers.Length];
            ActivationFunctionsDerivatives = new Action<float[], float[]>[Layers.Length];

            for (int i = 1; i < layers.Length; i++)
            {
                ActivationFunctions[i] = Activations.GetActivationFunction(activationFunctionsTypes[i - 1]);
                ActivationFunctionsDerivatives[i] = Activations.GetActivationDerivative(activationFunctionsTypes[i - 1]);
            }

        }

        public float[] FeedForward(float[] input, float[] output)
        {
            if (input.Length != Layers[0])
                throw new Exception("Input is not of right size");

            Buffer.BlockCopy(input, 0, Neurons[0], 0, sizeof(float) * Layers[0]); //copy input into Neurons (fast)

            for (int l = 1; l < Layers.Length; l++)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                    Z[l][n] = TensorPrimitives.Dot(Weights[l].AsSpan(Layers[l - 1] * n, Layers[l - 1]), Neurons[l - 1]) + Biases[l][n];

                ActivationFunctions[l](Z[l], Neurons[l]);
            }

            Buffer.BlockCopy(Neurons[Layers.Length - 1], 0, output, 0, sizeof(float) * Layers[Layers.Length - 1]);
            return output;
        }
    }
}
