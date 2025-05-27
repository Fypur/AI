using System.Text.Json;

namespace AI
{
    public class NN2 : Network
    {
        public float LearningRate { get; set; }
        public int[] Layers { get; set; }
        public float Beta1 { get; set; } = 0.9f;
        public float Beta2 { get; set; } = 0.999f;
        public long Timestep { get; set; } = 1;

        public float[][] Neurons;
        public float[][] Z;
        public float[][] Biases;
        public float[][][] Weights { get; set; }
        public float[][][] MovingAverage { get; set; }
        public float[][] MovingAverageBiases { get; set; }
        public float[][][] MovingSqrdAverage { get; set; }
        public float[][] MovingSqrdAverageBiases { get; set; }

        public Func<float, float> ActivationHidden { get; set; } = ELU;
        public Func<float, float> ActivationOut { get; set; } = LeakyReLU;
        public Func<float, float> ActivationHiddenDer;
        public Func<float, float> ActivationOutDer;

        private const float epsilon = 0.00000001f; //prevent division by zero

        private float[][] error;
        private float[][][] moveWeights;
        private float[][] moveBiases;
        private static JsonSerializerOptions jsonOptions = new() { WriteIndented = false } ;

        public NN2(int[] layers, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            Layers = layers;
            LearningRate = learningRate;
            Beta1 = beta1; //0.9f most of the time
            Beta2 = beta2; //0.999f most of the time

            /*if (ActivationHidden == Softmax)
                throw new Exception("Softmax hasn't been implemented to handle hidden layer usage");*/

            ActivationHiddenDer = Derivatives(ActivationHidden);
            ActivationOutDer = Derivatives(ActivationOut);

            //Init Everything
            Neurons = new float[Layers.Length][];
            Z = new float[Layers.Length][];
            Biases = new float[Layers.Length][];
            MovingAverageBiases = new float[Layers.Length][];
            MovingSqrdAverageBiases = new float[Layers.Length][];
            Weights = new float[Layers.Length][][];
            MovingAverage = new float[Layers.Length][][];
            MovingSqrdAverage = new float[Layers.Length][][];
            error = new float[Layers.Length][];

            Neurons[0] = new float[Layers[0]];
            Z[0] = new float[Layers[0]];
            error[0] = new float[Layers[0]];

            moveBiases = new float[Biases.Length][];
            moveWeights = new float[Weights.Length][][];


            for (int l = 1; l < Layers.Length; l++)
            {
                Neurons[l] = new float[Layers[l]];
                Z[l] = new float[Layers[l]];
                Biases[l] = new float[Layers[l]];
                MovingAverageBiases[l] = new float[Layers[l]];
                MovingSqrdAverageBiases[l] = new float[Layers[l]];
                Weights[l] = new float[Layers[l]][];
                MovingAverage[l] = new float[Layers[l]][];
                MovingSqrdAverage[l] = new float[Layers[l]][];

                error[l] = new float[Neurons[l].Length];
                moveBiases[l] = new float[Biases[l].Length];
                moveWeights[l] = new float[Weights[l].Length][];

                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    Biases[l][n] = GaussianRandom(0, 0.5f);


                    MovingAverageBiases[l][n] = 0;
                    MovingSqrdAverageBiases[l][n] = 0;
                    Weights[l][n] = new float[Layers[l - 1]];
                    MovingAverage[l][n] = new float[Layers[l - 1]];
                    MovingSqrdAverage[l][n] = new float[Layers[l - 1]];
                    moveWeights[l][n] = new float[Weights[l][n].Length];


                    float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                    for (int prevLayerN = 0; prevLayerN < Neurons[l - 1].Length; prevLayerN++)
                    {
                        Weights[l][n][prevLayerN] = GaussianRandom(0, std);
                        MovingAverage[l][n][prevLayerN] = 0;
                        MovingSqrdAverage[l][n][prevLayerN] = 0;
                    }
                }
            }
        }


        public float[] FeedForward(float[] input)
        {
            if (input.Length != Layers[0])
                throw new Exception("Input is not of right size");

            if (input.Contains(float.NaN))
                throw new Exception("Input contains NaN values");

            for (int i = 0; i < Neurons[0].Length; i++)
                Neurons[0][i] = input[i];

            for (int l = 1; l < Layers.Length; l++)
            {
                //Parallel.For(0, Neurons[l].Length, (n) =>
                for (int n = 0; n < Neurons[l].Length; n++)
                {

                    Z[l][n] = 0;

                    //Parallel.For(0, Neurons[l - 1].Length, (prevN) =>
                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)

                    {
                        Z[l][n] += Weights[l][n][prevN] * Neurons[l - 1][prevN];
                    }//);

                    Z[l][n] += Biases[l][n];


                    if (l != Layers.Length - 1)
                        Neurons[l][n] = ActivationHidden(Z[l][n]);
                    else
                        Neurons[l][n] = ActivationOut(Z[l][n]);
                }//);
            }

            float[] output = new float[Neurons[Layers.Length - 1].Length];
            for (int i = 0; i < Neurons[Layers.Length - 1].Length; i++)
                output[i] = Neurons[Layers.Length - 1][i];

            return output;
        }


        public void Train(float[][] inputs, float[][] targets)
        {
            float totCost = 0f;
            TrainLoss(inputs, (i, output) =>
            {
                for (int j = 0; j < output.Length; j++)
                {
                    output[j] = 2 * (targets[i][j] - output[j]);
                    totCost += (targets[i][j] - output[j]) * (targets[i][j] - output[j]);
                }
                return output;
            });
        }

        public void TrainLoss(float[][] inputs, Func<int, float[], float[]> lossFunction)
        {
            for (int p = 0; p < inputs.Length; p++)
            {
                float[] input = inputs[p];
                float[] loss = lossFunction(p, FeedForward(input));

                //Computing the error
                //The error is basically the derivative of the cost by the z of that neuron at that place
                for (int i = 0; i < Layers[Layers.Length - 1]; i++)
                    error[Neurons.Length - 1][i] = loss[i] * ActivationOutDer(Z[Layers.Length - 1][i]);

                for (int l = Layers.Length - 1; l >= 2; l--)
                {
                    error[l - 1] = new float[Neurons[l - 1].Length];
                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                    {
                        for (int n = 0; n < Neurons[l].Length; n++)
                            error[l - 1][prevN] += error[l][n] * Weights[l][n][prevN];

                        error[l - 1][prevN] *= ActivationHiddenDer(Z[l - 1][prevN]);
                    }
                }


                for (int l = 1; l < Layers.Length; l++)
                {
                    for (int n = 0; n < Neurons[l].Length; n++)
                    {
                        moveBiases[l][n] += error[l][n];

                        for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                            moveWeights[l][n][prevN] += error[l][n] * Neurons[l - 1][prevN];
                    }
                }
            }

            for (int l = 1; l < Layers.Length; l++)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    moveBiases[l][n] /= inputs.Length;

                    MovingAverageBiases[l][n] = Beta1 * MovingAverageBiases[l][n] + (1 - Beta1) * moveBiases[l][n];
                    MovingSqrdAverageBiases[l][n] = Beta2 * MovingSqrdAverageBiases[l][n] + (1 - Beta2) * moveBiases[l][n] * moveBiases[l][n];
                    float mHat = MovingAverageBiases[l][n] / (1f - (float)Math.Pow(Beta1, Timestep));
                    float vHat = MovingSqrdAverageBiases[l][n] / (1f - (float)Math.Pow(Beta2, Timestep));

                    Biases[l][n] += LearningRate * mHat / ((float)Math.Sqrt(vHat) + epsilon);

                    for (int prevN = 0; prevN < Neurons[l - 1].Length; prevN++)
                    {
                        moveWeights[l][n][prevN] /= inputs.Length;

                        MovingAverage[l][n][prevN] = Beta1 * MovingAverage[l][n][prevN] + (1 - Beta1) * moveWeights[l][n][prevN];
                        MovingSqrdAverage[l][n][prevN] = Beta2 * MovingSqrdAverage[l][n][prevN] + (1 - Beta2) * moveWeights[l][n][prevN] * moveWeights[l][n][prevN];

                        float mHatW = MovingAverage[l][n][prevN] / (1f - (float)Math.Pow(Beta1, Timestep));
                        float vHatW = MovingSqrdAverage[l][n][prevN] / (1f - (float)Math.Pow(Beta2, Timestep));

                        Weights[l][n][prevN] += LearningRate * mHatW / ((float)Math.Sqrt(vHatW) + epsilon);

                        moveWeights[l][n][prevN] = 0;
                    }

                    moveBiases[l][n] = 0;
                }
            }

            Timestep++;
        }

        public void CheckNetwork()
        {
            for (int l = 0; l < Layers.Length; l++)
            {
                Check(Neurons[l]);

                if (l != 0)
                {
                    Check(Biases[l]);

                    for (int n = 0; n < Neurons[l].Length; n++)
                        Check(Weights[l][n]);
                }
            }
        }

        #region Activations
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

        private static float LeakyReLU(float x)
        {
            if (x >= 0)
                return x;
            return 0.01f * x;
        }

        private static float LeakyReLUPrime(float x)
        {
            if (x > 0)
                return 1;
            return 0.01f;
        }

        private static float ELU(float x)
        {
            if (x >= 0)
                return x;
            return (float)Math.Exp(x) - 1;
        }

        private static float ELUPrime(float x)
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
            if (function == LeakyReLU)
                return LeakyReLUPrime;
            if (function == ELU)
                return ELUPrime;
            if (function == Linear)
                return LinearPrime;
            /*if (function == Softmax)
                return SoftmaxPrime;*/

            throw new Exception("Could not find derivative of Activation Function");
        }

        #endregion

        #region Checks

        private static void Check(float[] f)
        {
            for (int i = 0; i < f.Length; i++)
                Check(f[i]);
        }

        private static void Check(float f)
        {
            if (!float.IsNormal(f) && f != 0)
                throw new Exception("float has been checked as NaN or too big");
        }

        #endregion

        #region Utils

        //https://stackoverflow.com/questions/218060/random-gaussian-variables
        private static float GaussianRandom()
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal;
        }

        private static float GaussianRandom(float mean, float standardDeviation)
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + standardDeviation * randStdNormal; //random normal(mean,stdDev^2)
            return (float)randNormal;
        }

        public void Save(string filePath)
        {
            string s = JsonSerializer.Serialize(this, jsonOptions);
            filePath = filePath.Replace('\\', '/');
            if (!filePath.EndsWith(".json"))
                filePath += ".json";

            File.WriteAllText(filePath, s);
        }

        public void Load(string filePath)
        {
            filePath = filePath.Replace('\\', '/');
            if (!filePath.EndsWith(".json"))
                filePath += ".json";

            string s = File.ReadAllText(filePath);
            JsonSerializer.Deserialize<NN2>(s)!.CopyTo(this);
        }

        public void CopyTo(NN2 network)
        {
            int[] la = new int[Layers.Length];
            Array.Copy(Layers, la, Layers.Length);
            network.Layers = la;
            for (int l = 1; l < Layers.Length; l++)
            {
                for (int n = 0; n < Layers[l]; n++)
                {
                    network.Biases[l][n] = Biases[l][n];
                    network.MovingAverageBiases[l][n] = MovingAverageBiases[l][n];
                    network.MovingSqrdAverageBiases[l][n] = MovingSqrdAverageBiases[l][n];
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                    {
                        network.Weights[l][n][prevN] = Weights[l][n][prevN];
                        network.MovingAverage[l][n][prevN] = MovingAverage[l][n][prevN];
                        network.MovingSqrdAverage[l][n][prevN] = MovingSqrdAverage[l][n][prevN];
                    }
                }
            }
        }

        public Network Copy()
        {
            NN2 nn = new NN2(Layers, LearningRate, Beta1, Beta2);
            CopyTo(nn);

            return nn;
        }

        #endregion
    }
}
