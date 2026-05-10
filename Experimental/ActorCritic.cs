using System;
using System.Reflection;

namespace AI
{
    //Made with the help https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
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

        public void TrainOneEpisode(float[][] states, float[] rewards, int[] takenActions)
        {
            float[] adv2 = GetExpectedReturn(states, rewards);
            float[] advantages = GetStandardizedAdvantage(states, rewards);

            Network.TrainLoss(states, (index, output) => ComputeLoss(output, advantages[index], takenActions[index]));
        }

        public float[] ComputeLoss(float[] networkOutput, float advantage, int takenAction) //honestly the most difficult part
        {
            float[] softmax = Softmax(networkOutput, networkOutput.Length - 1);

            //float criticLoss = HuberLoss(networkOutput[networkOutput.Length - 1], expectedReturn); //we don't actually care about calculating the huber loss, tensorflow does it because it automatically differentiates it
            //Calculate differentiated Huber Loss
            //We just differentiate this https://en.wikipedia.org/wiki/Huber_loss
            //float advantage = expectedReturn - networkOutput[networkOutput.Length - 1];
            float d = 1;
            float differentiatedCriticLoss = Math.Abs(advantage) <= d ? advantage : d * Math.Sign(advantage);

            float[] differenciatedLoss = new float[networkOutput.Length];
            differenciatedLoss[differenciatedLoss.Length - 1] = differentiatedCriticLoss;

            for (int action = 0; action < networkOutput.Length - 1; action++)
                differenciatedLoss[action] = SoftmaxDer(softmax, takenAction, action) * advantage / softmax[takenAction];

            for(int i = 0; i < differenciatedLoss.Length; i++)
            {
                if (float.IsNaN(differenciatedLoss[i]))
                { }
            }
            
            return differenciatedLoss;
        }

        public float[] GetExpectedReturn(float[][] states, float[] rewards)
        {
            float[] returns = new float[rewards.Length];


            for (int i = 0; i < returns.Length; i++)
                returns[i] = rewards[i];

            float sum = 0;
            for (int j = rewards.Length - 1; j >= 0; j--)
            {
                float temp = sum;
                sum += rewards[j];
                returns[j] += temp;
                sum *= Gamma;
            }

            for (int i = 0; i < rewards.Length; i++)
                returns[i] -= Network.FeedForward(states[i])[actionSize];

            //standardize return
            float stddeviation = 0; //ecart type
            float mean = returns.Sum() / returns.Length;
            for (int i = 0; i < returns.Length; i++)
                stddeviation += (returns[i] - mean) * (returns[i] - mean);

            stddeviation = (float)Math.Sqrt(stddeviation / (returns.Length)) + 0.000000000001f;

            for (int i = 0; i < returns.Length; i++)
                returns[i] = (returns[i] - mean) / stddeviation;

            return returns;
        }

        public float[] GetStandardizedAdvantage(float[][] states, float[] rewards)
        {
            //LAMBDA RETURN
            float lambda = 0.95f;

            float[] returns = new float[rewards.Length];
            float[] values = new float[rewards.Length];
            float[] advantages = new float[rewards.Length];

            for(int i = 0; i < rewards.Length; i++)
                values[i] = Network.FeedForward(states[i])[actionSize];

            for (int t = 0; t < rewards.Length; t++)
            {
                float lambdaPowed = 1;
                for(int n = 1; n < rewards.Length - t; n++)
                {
                    returns[t] += lambdaPowed * NStepReturn(t, n, values[t + n]);
                    lambdaPowed *= lambda;
                }

                returns[t] *= 1 - lambda;
                returns[t] += lambdaPowed * NStepReturn(t, rewards.Length - t, 0);
            }

            for(int i = 0; i < rewards.Length; i++)
                advantages[i] = returns[i] - values[i];

            Standardize(advantages);

            return advantages;


            float NStepReturn(int index, int n, float valueNStepsFurther)
            {
                float nReturn = 0;
                float gammaPowed = 1;
                for (int i = index; i < Math.Min(i + n, rewards.Length); i++)
                {
                    nReturn += rewards[i] * gammaPowed;
                    gammaPowed *= Gamma;
                }

                nReturn += gammaPowed * valueNStepsFurther;
                return nReturn;
            }
        }

        public void Standardize(float[] data)
        {
            float stdDeviation = 0;
            float mean = data.Sum() / data.Length;
            for (int i = 0; i < data.Length; i++)
                stdDeviation += (data[i] - mean) * (data[i] - mean);

            stdDeviation = (float)Math.Sqrt(stdDeviation / data.Length) + 0.000000000001f;

            for (int i = 0; i < data.Length; i++)
                data[i] = (data[i] - mean) / stdDeviation;
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

        float SoftmaxDer(float[] softmax, int takenAction, int inFonctionOf){
            if(takenAction == inFonctionOf) return softmax[takenAction] * (1 - softmax[takenAction]);
            else return -softmax[takenAction] * softmax[inFonctionOf];
        }

        float HuberLoss(float output, float target, float d = 1)
        {
            if (Math.Abs(output - target) <= d) return 0.5f * (output - target) * (output - target);
            else return d * (Math.Abs(output - target) - 0.5f * d);
        }

        public int Act(float[] state)
        {
            float[] probs = Softmax(Network.FeedForward(state), actionSize);

            double random = Rand.NextDouble();
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
        