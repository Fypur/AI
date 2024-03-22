using System;
using Fiourp;

namespace AI
{
    public class PolicyGradient
    {
        public NN2 LogitsNetwork;
        public float Gamma = 0.99f;

        private int ActionSize => LogitsNetwork.Layers[LogitsNetwork.Layers.Length - 1];

        public PolicyGradient(int[] layers, float learningRate = 0.001f, float movingAverageBeta = 0.9f, float gamma=0.99f)
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

            Console.WriteLine("Trained");

            LogitsNetwork.TrainLoss(batch.States,
            (index, logits) =>
            {
                float[] loss = new float[ActionSize];
                float[] softmax = Softmax(logits);

                float logp = (float)Math.Log(softmax[batch.Actions[index]]);
                int otherAction = batch.Actions[index] == 0 ? 1: 0;

                loss[batch.Actions[index]] = -logp * (returns[index] - 10);
                loss[otherAction] = logp * (returns[index] - 10);

                return loss;
            });
        }

        float SoftmaxDer(float[] softmax, int i, int j){
            if(i == j) return softmax[i] * (1 - softmax[i]);
            else return softmax[i] * softmax[j];
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