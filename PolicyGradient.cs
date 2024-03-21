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

            int episodeIndex = 0;
            for(int i = 0; i < returns.Length; i++)
            {
                for(int j = episodeIndex; j < episodeIndex + batch.EpisodeLengths[i]; j++)
                {
                    float gammaPow = Gamma;
                    for(int k = j + 1; k < episodeIndex + batch.EpisodeLengths[i]; k++)
                    {
                        returns[j] += gammaPow * returns[k];
                        gammaPow *= Gamma;
                    }
                }

                episodeIndex += batch.EpisodeLengths[i];
            }

            LogitsNetwork.TrainLoss(batch.States,
            (index, output) =>
            {
                float[] loss = new float[ActionSize];

                for(int i = 0; i < loss.Length; i++)
                    loss[i] = (float)Math.Log(Softmax(output)[batch.Actions[index]]) * returns[index];
                
                return loss;
            });
        }

        public int Act(float[] state)
        {
            float[] probs = Softmax(LogitsNetwork.FeedForward(state));

            float random = Rand.NextDouble();
            int i = 0;
            float sum = 0;
            for(; i < probs.Length; i++){
                sum += probs[i];
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