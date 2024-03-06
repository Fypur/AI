using System;
using Fiourp;

namespace AI
{
    public class DeepQAgent2
    {
        public NN2 Network;
        public NN2 TargetNetwork;

        public PrioritizedExperienceReplay2 ReplayBuffer;
        public float Epsilon { get; private set; } = 1;
        public float Gamma { get; private set; }
        public int TotalTimeSteps { get; private set; } //TODO: BETA DECAY TO 1 WITH TOTALTIMESTEPS
        public int TargetRefreshRate { get; private set; }
        public bool TrainingStarted { get; private set; }

        public ScottPlot.Plot LossPlot;
        public ScottPlot.Plottables.DataLogger DataLogger;

        private int step;
        private float epsilonMin;
        private float epsilonDecay;
        private bool saveMemory;
        private string saveFile;
        private int actionSize => Network.Layers[Network.Layers.Length - 1];

        
        public DeepQAgent2(int[] layers, int totalTimesteps = 100000, int targetRefreshRate = 1000, float learningRate = 0.001f, float movingAverageBeta = 0.9f,
         int memorySize = 65536, float replayAlpha = 0.6f, float replayBeta = 0.4f, float replayBetaIncrease = 0.000001f, float replayEpsilon = 0.000001f,
         int batchSize = 64, float epsilonMin = 0.03f, float epsilonDecay = 0.0001f, float gamma = 0.99f, bool saveMemory = true, string saveFile = "./memory")
        {
            ReplayBuffer = new PrioritizedExperienceReplay2(memorySize, replayAlpha, replayBeta, replayBetaIncrease, replayEpsilon, batchSize);
            Network = new NN2(layers, learningRate, movingAverageBeta);
            TargetNetwork = Network.Copy();

            Gamma = gamma;
            Epsilon = 1;
            TotalTimeSteps = totalTimesteps;
            TargetRefreshRate = targetRefreshRate;
            this.epsilonMin = epsilonMin;
            this.epsilonDecay = epsilonDecay;

            this.saveMemory = saveMemory;
            this.saveFile = saveFile;

            LossPlot = new ScottPlot.Plot();
            DataLogger = LossPlot.Add.DataLogger();
        }

        public int Act(float[] state)
        {
            if(!TrainingStarted)
                return Rand.NextInt(0, actionSize);

            Epsilon = epsilonMin + (1 - epsilonMin) * (float)Math.Exp(-epsilonDecay * step);

            if (Rand.NextDouble() < Epsilon)
                return Rand.NextInt(0, actionSize);

            return ArgMax(Network.FeedForward(state));
        }

        public void Replay()
        {
            if (!TrainingStarted)
                return;

            float totalError = 0;

            Sample sample = ReplayBuffer.Sample();
            float[][] inputs = new float[sample.Experience.Length][];
            int[] nextStateArgmax = new int[sample.Experience.Length];

            for(int i = 0; i < sample.Experience.Length; i++)
            {
                inputs[i] = sample.Experience[i].State;
                nextStateArgmax[i] = ArgMax(Network.FeedForward(sample.Experience[i].NextState));
            }

            Network.TrainLoss(inputs, GetLoss);

            float[] GetLoss(int bacthIndex, float[] networkOutput)
            {
                Experience exp = sample.Experience[bacthIndex];
                float weight = sample.Weights[bacthIndex];
                int replayIndex = sample.Indices[bacthIndex];

                float tdError;
                if (exp.Done)
                    tdError = exp.Reward;
                else
                    tdError = exp.Reward + Gamma * TargetNetwork.FeedForward(exp.NextState)[nextStateArgmax[bacthIndex]];

                tdError -= networkOutput[exp.Action];

                ReplayBuffer.ChangePriority(replayIndex, Math.Abs(tdError));

                float[] loss = new float[networkOutput.Length];
                loss[exp.Action] = tdError * weight;

                totalError += tdError * tdError;

                return loss;
            }

            DataLogger.Add(totalError / ReplayBuffer.BatchSize);
            step++;

            if (step % TargetRefreshRate == 0)
                RefreshTargetNetwork();
        }

        public void Remember(Experience experience)
        {
            ReplayBuffer.Add(experience);

            if (ReplayBuffer.Filled && !TrainingStarted)
            {
                if (saveMemory)
                    ReplayBuffer.Save(saveFile);

                TrainingStarted = true;
            }
        }

        public int ArgMax(float[] array)
        {
            float max = array[0];
            int index = 0;
            for(int i = 1; i < array.Length; i++)
            {
                if(array[i] > max)
                {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        }

        private void RefreshTargetNetwork()
            => TargetNetwork = Network.Copy();

        public void ExportLossGraph()
        {
            LossPlot.SavePng(System.Environment.CurrentDirectory + "/plot.png", 1000, 1000);
            Console.WriteLine("Loss Graph exported step: " + step);
        }

        public void ClearLossGraph()
        {
            LossPlot = new();
            DataLogger = LossPlot.Add.DataLogger();
            Console.WriteLine("Cleared Loss Graph: " + step);
        }

        public void Save(string directoryPath)
        {
            directoryPath = directoryPath.Replace('\\', '/');
            if (directoryPath.EndsWith('/'))
                directoryPath = directoryPath.Substring(0, directoryPath.Length - 1);

            Network.Save(directoryPath);

            directoryPath += '/';
            ReplayBuffer.Save(directoryPath + "memory");
            Drawing.DebugForever.Add("SAVED");
        }

        public void Load(string directoryPath)
        {
            directoryPath = directoryPath.Replace('\\', '/');
            if (!directoryPath.EndsWith('/')) directoryPath += '/';

            Network.Load(directoryPath);
            ReplayBuffer.Load(directoryPath + "memory");
            Drawing.DebugForever.Add("LOADED");

            TrainingStarted = true;
            epsilonDecay = 100000;
        }
    }
}
