using System;
using Fiourp;

namespace AI
{
    public class PolicyGradient
    {
        public NN2 LogitsNetwork;
        public float Gamma = 0.99f;

        private int ActionSize => LogitsNetwork.Layers[LogitsNetwork.Layers.Length - 1];

        public PolicyGradient(int[] layers, float learningRate = 0.01f, float movingAverageBeta = 0.9f, float gamma=0.99f)
        {
            LogitsNetwork = new NN2(layers, learningRate, movingAverageBeta);
            Gamma = gamma;
        }

        public void Train(PolicyGradientBatch batch)
        {
            float[] returns = new float[batch.Rewards.Length];
            for(int i = 0; i < returns.Length; i++)
                returns[i] = batch.Rewards[i];

            int episodeIndex = 0;
            for(int i = 0; i < batch.EpisodeLengths.Length; i++)
            {
                float sum = 0;
                for(int j = episodeIndex + batch.EpisodeLengths[i] - 1; j >= episodeIndex; j--){
                    float temp = sum;
                    sum += returns[j];
                    returns[j] += temp;
                    sum *= Gamma;
                }

                episodeIndex += batch.EpisodeLengths[i];
            }

            //standardize return
            float stddeviation = 0; //ecart type
            float mean = returns.Sum() / returns.Length;
            for(int i = 0; i < returns.Length; i++)
                stddeviation += (returns[i] - mean) * (returns[i] - mean);

            stddeviation = (float)Math.Sqrt(stddeviation / returns.Length - 1) + 0.000000000001f;

            for (int i = 0; i < returns.Length; i++)
                returns[i] = (returns[i] - mean) / stddeviation;


            LogitsNetwork.TrainLoss(batch.States,
            (index, logits) =>
            {
                float[] outputLoss = new float[ActionSize];
                float[] loss = new float[ActionSize];
                float[] softmax = Softmax(logits);
                int otherAction = batch.Actions[index] == 0 ? 1: 0;

                //Gotta do some softmax backprop

                float logp = (float)Math.Log(softmax[batch.Actions[index]]);

                loss[batch.Actions[index]] = -logp * returns[index];

                for (int k = 0; k < ActionSize; k++)
                    for (int i = 0; i < ActionSize; i++)
                        outputLoss[i] += SoftmaxDer(softmax, k, i) * loss[k];

                return outputLoss;
            });

            Console.WriteLine("Trained");
        }

        float SoftmaxDer(float[] softmax, int inputIndex, int outputIndex){
            if(inputIndex == outputIndex) return softmax[inputIndex] * (1 - softmax[inputIndex]);
            else return -softmax[inputIndex] * softmax[outputIndex];
        }

        public int Act(float[] state)
        {
            float[] probs = Softmax(LogitsNetwork.FeedForward(state));

            float random = Rand.NextDouble();
            int i = 0;
            float sum = 0;
            for(; i < probs.Length; i++){
                sum += probs[i];
                if(i == probs.Length - 1) sum = 1.1f;
                if(sum >= random)
                    break;
            }

            return i;
        }

        /*public double LogLikelihood(float[] state, int action)
            => Math.Log(Softmax(LogitsNetwork.FeedForward(state))[action]);*/

        public float[] Softmax(float[] input)
        {
            float[] r = new float[input.Length];
            float sum = 0;

            for(int i = 0; i < r.Length; i++)
            {
                r[i] = (float)Math.Exp(input[i]);
                sum += r[i];
            }

            for(int i = 0; i < r.Length; i++)
                r[i] /= sum;

            return r;
        }
    }

    public class PolicyGradientBatch
        {
            //Episode -> TimeStep -> value
            public float[][] States;
            public float[] Rewards;
            public int[] Actions;
            public int[] EpisodeLengths;

            public PolicyGradientBatch(float[][] states, float[] rewards, int[] actions, int[] episodeLengths)
            {
                States = states;
                Rewards = rewards;
                Actions = actions;
                EpisodeLengths = episodeLengths;
            }
        }
}