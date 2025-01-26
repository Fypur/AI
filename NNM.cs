using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AI
{
    public class NNM : Network
    {
        public int[] Layers;

        public float[][][] Neurons;
        public float[][][] Z;
        public float[][] Biases;
        public float[][][] Weights;
        public float[][][] MovingAverage;
        public float[][] MovingAverageBiases;

        public float LearningRate { get; set; }
        public static Func<float, float> ActivationHidden = eLU;
        public static Func<float, float> ActivationOut = Linear;

        public static Func<float, float> ActivationHiddenDer = Derivatives(ActivationHidden);
        public static Func<float, float> ActivationOutDer = Derivatives(ActivationOut);

        public float Beta { get; set; } = 0.9f;

        private float[][][] errors;
        private float[][][][] moveWeights;
        private float[][][] finalMoveWeights;
        private float[][][] moveBiases;
        private float[][] finalMoveBiases;

        public int NThread = 10;
        private float[][] inputs;

        private Mutex[][][] mWeights;
        private Mutex[][] mBiases;
        private Func<int, float[], float[]> lossFunction;

        public NNM(int[] layers, float learningRate, float movingAverageBeta, int nThreads) : base()
        {
            NThread = nThreads;
            Layers = layers;
            LearningRate = learningRate;
            Beta = movingAverageBeta; //0.9f most of the time

            //Init Everything
            Biases = new float[Layers.Length][];
            mBiases = new Mutex[Layers.Length][];
            MovingAverageBiases = new float[Layers.Length][];
            Weights = new float[Layers.Length][][];
            mWeights = new Mutex[Layers.Length][][];
            MovingAverage = new float[Layers.Length][][];


            Neurons = new float[NThread][][];
            Z = new float[NThread][][];
            errors = new float[NThread][][];
            moveBiases = new float[NThread][][];
            moveWeights = new float[NThread][][][];

            finalMoveBiases = new float[Biases.Length][];
            finalMoveWeights = new float[Weights.Length][][];

            for (int i = 0; i < NThread; i++)
            {
                Neurons[i] = new float[Layers.Length][];
                Neurons[i][0] = new float[Layers[0]];
                Z[i] = new float[Layers.Length][];
                Z[i][0] = new float[Layers[0]];

                moveBiases[i] = new float[Biases.Length][];
                moveWeights[i] = new float[Weights.Length][][];
                errors[i] = new float[Layers.Length][];
                errors[i][0] = new float[Layers[0]];
            }


            for (int l = 1; l < Layers.Length; l++)
            {
                Biases[l] = new float[Layers[l]];
                mBiases[l] = new Mutex[Layers[l]];
                MovingAverageBiases[l] = new float[Layers[l]];
                Weights[l] = new float[Layers[l]][];
                mWeights[l] = new Mutex[Layers[l]][];
                MovingAverage[l] = new float[Layers[l]][];

                finalMoveBiases[l] = new float[Biases[l].Length];
                finalMoveWeights[l] = new float[Weights[l].Length][];

                for (int t = 0; t < NThread; t++)
                {
                    Neurons[t][l] = new float[Layers[l]];
                    Z[t][l] = new float[Layers[l]];
                    errors[t][l] = new float[Layers[l]];
                    moveBiases[t][l] = new float[Biases[l].Length];
                    moveWeights[t][l] = new float[Weights[l].Length][];
                }

                for (int n = 0; n < Layers[l]; n++)
                {
                    Biases[l][n] = GaussianRandom(0, 0.5f);
                    mBiases[l][n] = new Mutex();


                    MovingAverageBiases[l][n] = 1;
                    Weights[l][n] = new float[Layers[l - 1]];
                    mWeights[l][n] = new Mutex[Layers[l - 1]];
                    MovingAverage[l][n] = new float[Layers[l - 1]];

                    finalMoveWeights[l][n] = new float[Weights[l][n].Length];
                    for (int t = 0; t < NThread; t++)
                        moveWeights[t][l][n] = new float[Weights[l][n].Length];


                    float std = (float)Math.Sqrt(2.0 / Layers[l - 1]);

                    for (int prevLayerN = 0; prevLayerN < Layers[l - 1]; prevLayerN++)
                    {
                        Weights[l][n][prevLayerN] = GaussianRandom(0, std);
                        mWeights[l][n][prevLayerN] = new Mutex();
                        MovingAverage[l][n][prevLayerN] = 1;
                    }
                }
            }
        }

        public float[] FeedForward(float[] input)
            => FeedForward(input, 0);
        private float[] FeedForward(float[] input, int id)
        {
            if (input.Length != Layers[0])
                throw new Exception("Input is not of right size");

            if (input.Contains(float.NaN))
                throw new Exception("Input contains NaN values");

            for (int i = 0; i < Layers[0]; i++)
                Neurons[id][0][i] = input[i];

            for(int l = 1; l < Layers.Length; l++)
            {
                //Parallel.For(0, Layers[l], (n) =>
                for (int n = 0; n < Layers[l]; n++)
                {

                    Z[id][l][n] = 0;

                    //Parallel.For(0, Layers[l - 1], (prevN) =>
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)

                    {
                        Z[id][l][n] += Weights[l][n][prevN] * Neurons[id][l - 1][prevN];
                    }//);

                    Z[id][l][n] += Biases[l][n];


                    if (l != Layers.Length - 1)
                        Neurons[id][l][n] = ActivationHidden(Z[id][l][n]);
                    else
                        Neurons[id][l][n] = ActivationOut(Z[id][l][n]);
                }//);
            }

            float[] output = new float[Layers[Layers.Length - 1]];
            for (int i = 0; i < Neurons[id][Layers.Length - 1].Length; i++)
                output[i] = Neurons[id][Layers.Length - 1][i];

            return output;
        }
        public void Train(float[][] inputs, float[][] targets)
        {
            float[] Loss(int p, float[] output)
            {
                float[] l = new float[output.Length];
                for (int i = 0; i < l.Length; i++)
                    l[i] = 2 * (targets[p][i] - output[i]);
                return l;
            }

            TrainLoss(inputs, Loss);
        }

        public void TrainLoss(float[][] inputs, Func<int, float[], float[]> lossFunction)
        {
            this.inputs = inputs;
            this.lossFunction = lossFunction;
            Thread[] t = new Thread[NThread];

            for(int i = 0; i < NThread; i++)
            {
                int j = i;
                t[i] = new Thread(() => TrainT(j)); //Use j or else it breaks (using &i instead of !i)

            }

            for (int i = 0; i < t.Length; i++)
                t[i].Start();
            for (int i = 0; i < t.Length; i++)
                t[i].Join();

            for (int l = 1; l < Layers.Length; l++)
            {
                for (int n = 0; n < Layers[l]; n++)
                {
                    finalMoveBiases[l][n] /= inputs.Length;

                    MovingAverageBiases[l][n] = Beta * MovingAverageBiases[l][n] + (1 - Beta) * finalMoveBiases[l][n] * finalMoveBiases[l][n];
                    Biases[l][n] += finalMoveBiases[l][n] * (LearningRate / (float)Math.Sqrt(MovingAverageBiases[l][n]));

                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                    {
                        finalMoveWeights[l][n][prevN] /= inputs.Length;

                        MovingAverage[l][n][prevN] = Beta * MovingAverage[l][n][prevN] + (1 - Beta) * finalMoveWeights[l][n][prevN] * finalMoveWeights[l][n][prevN];
                        Weights[l][n][prevN] += finalMoveWeights[l][n][prevN] * (LearningRate / (float)Math.Sqrt(MovingAverage[l][n][prevN]));

                        finalMoveWeights[l][n][prevN] = 0;
                    }

                    finalMoveBiases[l][n] = 0;
                }
            }
        }

        private void TrainT(int id)
        {
            int start = (int)(inputs.Length / (float)NThread * id);
            int end = Math.Min((int)(inputs.Length / (float)NThread * (id + 1)), inputs.Length);

            for (int p = start; p < end; p++)
            {
                float[] input = inputs[p];
                float[] loss = lossFunction(p, FeedForward(input));
                float[] output = FeedForward(input, id);

                //Computing the error
                //The error is basically the derivative of the cost by the z of that neuron at that place
                for (int i = 0; i < Layers[Layers.Length - 1]; i++)
                    errors[id][Layers.Length - 1][i] = loss[i] * ActivationOutDer(Z[id][Layers.Length - 1][i]);

                for (int l = Layers.Length - 1; l >= 2; l--)
                {
                    errors[id][l - 1] = new float[Layers[l - 1]];
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                    {
                        for (int n = 0; n < Layers[l]; n++)
                            errors[id][l - 1][prevN] += errors[id][l][n] * Weights[l][n][prevN];

                        errors[id][l - 1][prevN] *= ActivationHiddenDer(Z[id][l - 1][prevN]);
                    }
                }


                for (int l = 1; l < Layers.Length; l++)
                {
                    for (int n = 0; n < Layers[l]; n++)
                    {
                        moveBiases[id][l][n] += errors[id][l][n];

                        for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                            moveWeights[id][l][n][prevN] += errors[id][l][n] * Neurons[id][l - 1][prevN];
                    }
                }
            }


            int mid = (int)(Layers.Length / (float)NThread * id) + 1;
            for (int l = mid; l < Layers.Length; l++)
            {
                for (int n = 0; n < Layers[l]; n++)
                {

                    mBiases[l][n].WaitOne();
                    finalMoveBiases[l][n] += moveBiases[id][l][n];
                    mBiases[l][n].ReleaseMutex();

                    moveBiases[id][l][n] = 0;

                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                    {

                        mWeights[l][n][prevN].WaitOne();

                        finalMoveWeights[l][n][prevN] += moveWeights[id][l][n][prevN];

                        mWeights[l][n][prevN].ReleaseMutex();

                        moveWeights[id][l][n][prevN] = 0;
                    }

                }
            }

            for (int l = 1; l < mid; l++)
            {
                for (int n = 0; n < Layers[l]; n++)
                {

                    mBiases[l][n].WaitOne();
                    finalMoveBiases[l][n] += moveBiases[id][l][n];
                    mBiases[l][n].ReleaseMutex();

                    moveBiases[id][l][n] = 0;

                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                    {

                        mWeights[l][n][prevN].WaitOne();

                        finalMoveWeights[l][n][prevN] += moveWeights[id][l][n][prevN];

                        mWeights[l][n][prevN].ReleaseMutex();

                        moveWeights[id][l][n][prevN] = 0;
                    }

                }
            }
        }

        //public void TrainLoss(float[][] inputs, Func<int, float[], float[]> lossFunction)

        public void CheckNetwork()
        {
            for (int l = 0; l < Layers.Length; l++)
            {
                for(int t = 0; t < NThread; t++)
                    Check(Neurons[t][l]);

                if(l != 0)
                {
                    Check(Biases[l]);

                    for (int n = 0; n < Layers[l]; n++)
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

        #endregion

        #region Checks

        void Check(float[] f)
        {
            for (int i = 0; i < f.Length; i++)
                Check(f[i]);
        }

        void Check(float f)
        {
            if (!float.IsNormal(f) && f != 0)
                throw new Exception("float has been checked as NaN or too big");
        }

        #endregion

        #region Utils

        //https://stackoverflow.com/questions/218060/random-gaussian-variables
        private float GaussianRandom()
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal;
        }

        private float GaussianRandom(float mean, float standardDeviation)
        {
            double u1 = 1.0 - Rand.NextDouble(); //uniform(0,1] random doubles/home/f/Documents/CarDeepQ/saves/netweights
            double u2 = 1.0 - Rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + standardDeviation * randStdNormal; //random normal(mean,stdDev^2)
            return (float)randNormal;
        }

        public void Save(string outputDir)
        {
            outputDir = outputDir.Replace('\\', '/');
            if (Directory.Exists(outputDir.Substring(0, outputDir.LastIndexOf('/'))))
            {
                Directory.CreateDirectory(outputDir);
                outputDir += "/";
            }
            else
                throw new Exception("Parent Dir does not exist");

            string jsonW = JsonSerializer.Serialize(this.Weights);
            string jsonB = JsonSerializer.Serialize(this.Biases);
            string jsonM = JsonSerializer.Serialize(this.MovingAverage);
            string jsonMB = JsonSerializer.Serialize(this.MovingAverageBiases);
            File.WriteAllText(outputDir + "weights.json", jsonW);
            File.WriteAllText(outputDir + "biases.json", jsonB);
            File.WriteAllText(outputDir + "movingAverage.json", jsonM);
            File.WriteAllText(outputDir + "movingAverageBiases.json", jsonMB);
        }

        public void Load(string inputDir)
        {
            inputDir = inputDir.Replace('\\', '/');
            if (inputDir.Last() != '/') inputDir += '/';

            string jsonW = File.ReadAllText(inputDir + "weights.json");
            string jsonB = File.ReadAllText(inputDir + "biases.json");
            /*NeuralNetwork n = JsonSerializer.Deserialize<NeuralNetwork>(json);

            LearningRate = n.LearningRate;
            weights = n.weights;
            biases = n.biases;*/


#pragma warning disable 324
            Weights = JsonSerializer.Deserialize<float[][][]>(jsonW);
            Biases = JsonSerializer.Deserialize<float[][]>(jsonB);
            MovingAverage = JsonSerializer.Deserialize<float[][][]>(File.ReadAllText(inputDir + "movingAverage.json"));
            MovingAverageBiases = JsonSerializer.Deserialize<float[][]>(File.ReadAllText(inputDir + "movingAverageBiases.json"));
#pragma warning restore 324
        }

        public Network Copy()
        {

            float[][][] w = new float[Layers.Length][][];
            float[][] b = new float[Layers.Length][];

            for (int l = 1; l < Layers.Length; l++)
            {
                b[l] = Biases[l];
                w[l] = new float[Layers[l]][];

                for (int n = 0; n < Layers[l]; n++)
                {
                    w[l][n] = new float[Layers[l - 1]];
                    for (int prevN = 0; prevN < Layers[l - 1]; prevN++)
                        w[l][n][prevN] = Weights[l][n][prevN];
                }
            }

            NNM neural = new NNM(Layers, LearningRate, Beta, NThread);
            neural.Weights = w;
            neural.Biases = b;
            return neural;
        }

        private static float NextFloat(float min, float max)
            => (float)(Rand.NextDouble() * (max - min) + min);

        #endregion
    }
}
