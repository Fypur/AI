using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Fiourp;

namespace AI;

//Helped heavily by https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
public class DeepQAgent
{
    //The values here are all default settings
    public float LearningRate = 0.001f;
    public float Gamma = 0.99f;
    private int actionSize = 4;
    private int[] layers;

    public float Epsilon { get; private set; } = 1;
    public float EpsilonMin = 0.03f;
    public float EpsilonDecay = 0.0001f;
    private float decayStep = 0;

    public int targetRefreshRate = 300;

    public int MemorySize = 65536; //or 65536 or 131072
    public int BatchSize = 128;

    public float alpha = 0.6f;
    public float beta = 0.4f;
    public float betaIncreasePerStep = 0.0000001f;
    public float betaMax = 0.9f;
    public float baseProbability = 0.001f;

    private PrioritizedExperienceReplay replayBuffer;
    private bool saveNewMemory;
    private bool loadModel;
    private bool learning;
    
    public Network Network { get; private set; }
    public Network TargetNetwork { get; private set; }

    private ScottPlot.Plot plot = new();
    private ScottPlot.Plottables.DataLogger dataLogger;

    private int step = 0;

    public string saveFile = System.IO.Directory.GetCurrentDirectory().Replace('\\', '/') + "/savedMem.json";

    public DeepQAgent(int[] layers, bool saveNewMemory, bool learning, bool loadModel)
    //We don't do anything here because we let the constructor set the parameters (if they don't touch anything they're using default parameters)
    {
        this.layers = layers;
        actionSize = layers[layers.Length - 1];
        this.learning = learning;
        this.saveNewMemory = saveNewMemory;
        this.loadModel = loadModel;
    }

    public void Init()
    {
        Network = new NN2(layers, LearningRate, 0.9f);
        TargetNetwork = Network.Copy();

        if (!learning)
        {
            Epsilon = 0;
            EpsilonDecay = 1;
        }

        if (!saveNewMemory)
        {
            replayBuffer = new(0);
            replayBuffer.Load(saveFile);
        }
        else
            replayBuffer = new(MemorySize);

        if (loadModel)
        {
            Load(System.IO.Directory.GetCurrentDirectory() + "/network");
            decayStep = 200000;
            Drawing.DebugForever.Add("LOADED");
        }

        dataLogger = plot.Add.DataLogger();
    }

    //This is where we train the algorithm
    public void Replay()
    {
        if (!replayBuffer.Filled)
            return;

        //beta = Math.Min(beta + betaIncreasePerStep, betaMax);
        Tuple<Experience, int>[] miniBatch = replayBuffer.Sample(MemorySize);
        if(saveNewMemory)
        {
            saveNewMemory = false;
            replayBuffer.Save(saveFile);
        }

        float error = 0;

        float[][] inputs = new float[miniBatch.Length][];
        int[] nextStateArgmax = new int[miniBatch.Length];

        for (int i = 0; i < miniBatch.Length; i++)
            inputs[i] = miniBatch[i].Item1.State;
        for (int i = 0; i < miniBatch.Length; i++)
            nextStateArgmax[i] = Argmax(Network.FeedForward(miniBatch[i].Item1.NextState));

        float maxWeight = (float)Math.Pow(BatchSize * replayBuffer.MinProbability() / replayBuffer.ProbabilitySum(), -beta);

        Network.TrainLoss(inputs, (i, outputCurrentState) =>
        {
            Experience exp = miniBatch[i].Item1;
            int index = miniBatch[i].Item2;
            float[] loss = new float[outputCurrentState.Length];

            float target;
            if (exp.Done)
                target = exp.Reward; //if we are on a terminal state
            else
                target = exp.Reward + Gamma * TargetNetwork.FeedForward(exp.NextState)[nextStateArgmax[i]];

            loss[exp.Action] = target - outputCurrentState[exp.Action];

            float oldProbability = replayBuffer.GetPriority(index);
            float weight = (float)Math.Pow(MemorySize * oldProbability / replayBuffer.ProbabilitySum(), -beta);


            weight /= maxWeight; //normalization

            float newProbability = (float)Math.Pow(Math.Abs(loss[exp.Action]) + baseProbability, alpha);
            replayBuffer.UpdatePriority(index, newProbability); //we update the priority according to TD error

            loss[exp.Action] *= weight; //importance sampling weights to counterbalance bias
            error += loss[exp.Action] * loss[exp.Action];

            return loss; 
        });

        dataLogger.Add(error / BatchSize);
        step += 1;

        if (step % 1000 == 0)
            replayBuffer.UpdateMaxPriority();            
    }

    public int Argmax(float[] array){
        int argMax = 0;
        float max = array[0];
        for (int k = 1; k < array.Length; k++)
            if (array[k] > max)
            {
                max = array[k];
                argMax = k;
            }
        return argMax;
    }

    public void Remember(float[] state, int action, float reward, float[] nextState, bool done)
    {
        replayBuffer.Add(new Experience(state, action, reward, nextState, done), replayBuffer.MaxPriority);
    }

    public int Act(float[] state)
    {
        if (!replayBuffer.Filled)
            return Rand.NextInt(0, actionSize);

        decayStep += 1f;

        Epsilon = EpsilonMin + (1 - EpsilonMin) * (float)Math.Exp(-EpsilonDecay * decayStep);
        //epsilon -= 0.001f;
        var r = Rand.NextDouble();
        //r = 1;
        //r = 1;
        if (r < Epsilon)
        {
            int r2 = Rand.NextInt(0, actionSize);
            return r2;
        }
        float[] netValues = Network.FeedForward(state);
        float max = netValues[0];
        int argMax = 0;
        for(int i = 1; i < netValues.Length; i++)
        {
            if (netValues[i] > max)
            {
                max = netValues[i];
                argMax = i;
            }
        }

        return argMax;
    }

    public void RefreshTargetNetwork()
        => TargetNetwork = Network.Copy();

    public bool HasStartedTraining()
        => replayBuffer.Filled;

    public void ExportLossGraph()
    {
        plot.SavePng(System.Environment.CurrentDirectory + "/plot.png", 1000, 1000);
        Console.WriteLine("Loss Graph exported step: " + step);
    }

    public void ClearLossGraph()
    {
        plot = new();
        dataLogger = plot.Add.DataLogger();
        Console.WriteLine("Cleared Loss Graph: " + step);
    }

    public void Save(string directoryPath)
    {
        directoryPath = directoryPath.Replace('\\', '/');
        if (directoryPath.EndsWith('/'))
            directoryPath = directoryPath.Substring(0, directoryPath.Length - 1);

        Network.Save(directoryPath);

        directoryPath += '/';
        replayBuffer.Save(directoryPath + "memory");
        Drawing.DebugForever.Add("SAVED");
    }

    public void Load(string directoryPath)
    {
        directoryPath = directoryPath.Replace('\\', '/');
        if (!directoryPath.EndsWith('/')) directoryPath += '/';

        Network.Load(directoryPath);
        replayBuffer.Load(directoryPath + "memory");
        Drawing.DebugForever.Add("LOADED");
    }
}
