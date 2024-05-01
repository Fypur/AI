using System;
using Fiourp;

namespace AI
{
    public class ActorCritic
    {
        public NN2 Network;
        public float Gamma = 0.99f;

        private int actionSize => Network.Layers[Network.Layers.Length - 1] - 1;

        public ActorCritic(int[] layers, float learningRate = 0.01f, float movingAverageBeta = 0.9f, float gamma=0.99f)
        {
            Network = new NN2(layers, learningRate, movingAverageBeta);
            Gamma = gamma;
        }

        public float[] GetExpectedReturn(float[] rewards)
        {
            float[] returns = new float[rewards.Length];

            float sum = 0;
            for(int j = returns.Length - 1; j >= 0; j--){
                sum += rewards[j];
                returns[j] += sum;
                sum *= Gamma;
            }

            //standardize return
            float stddeviation = 0; //ecart type
            float mean = returns.Sum() / returns.Length;
            for(int i = 0; i < returns.Length; i++)
                stddeviation += (returns[i] - mean) * (returns[i] - mean);

            stddeviation = (float)Math.Sqrt(stddeviation / returns.Length) + 0.000000000001f;

            for (int i = 0; i < returns.Length; i++)
                returns[i] = (returns[i] - mean) / stddeviation;

            return returns;
        }

        public float[] Softmax(float[] input, int length)
        {
            float[] r = new float[length];
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

        float SoftmaxDer(float[] softmax, int inputIndex, int outputIndex){
            if(inputIndex == outputIndex) return softmax[inputIndex] * (1 - softmax[inputIndex]);
            else return -softmax[inputIndex] * softmax[outputIndex];
        }

        public int Act(float[] state)
        {
            float[] probs = Softmax(Network.FeedForward(state), actionSize);

            float random = Rand.NextDouble();
            int i = 0;
            float sum = 0;
            for(; i < probs.Length - 1; i++){
                sum += probs[i];
                if(sum >= random)
                    break;
            }

            return i;
        }
    }
}
        