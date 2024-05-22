using System;
using Fiourp;

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
            float[] returns = GetExpectedReturn(rewards);

            Network.TrainLoss(states, (index, output) => ComputeLoss(output, returns[index], takenActions[index]));
        }

        public float[] ComputeLoss(float[] networkOutput, float expectedReturn, int takenAction) //honestly the most difficult part
        {
            float[] softmax = Softmax(networkOutput, networkOutput.Length - 1);

            //float criticLoss = HuberLoss(networkOutput[networkOutput.Length - 1], expectedReturn); //we don't actually care about calculating the huber loss, tensorflow does it because it automatically differentiates it
            //Calculate differentiated Huber Loss
            //We just differentiate this https://en.wikipedia.org/wiki/Huber_loss
            float advantage = expectedReturn - networkOutput[networkOutput.Length - 1];
            float d = 1;
            float differentiatedCriticLoss = Math.Abs(advantage) <= advantage ? advantage : d * Math.Sign(advantage);

            float[] differenciatedLoss = new float[networkOutput.Length];
            differenciatedLoss[differenciatedLoss.Length - 1] = differentiatedCriticLoss;

            for (int action = 0; action < networkOutput.Length - 1; action++)
                differenciatedLoss[action] = SoftmaxDer(softmax, takenAction, action) * advantage / softmax[takenAction];
            
            return differenciatedLoss;
        }

        public float[] GetExpectedReturn(float[] rewards)
        {
            //LAMBDA RETURN
            float lambda = 0.95f;
            float[] returns = new float[rewards.Length];

            float sum = 0;
            for(int j = returns.Length - 1; j >= 0; j--){
                sum += rewards[j];
                returns[j] += sum;
                sum *= Gamma;
            }

            for(int t = 0; t < rewards.Length; t++){
                float added = 0;
                float lambdaPowed = 1;
                for(int n = 1; n < rewards.Length; n++){

                    //Calculate n step return
                    float nStepReturn = 0;
                    float gammaPow = 1;
                    for(int i = 0; i < n; i++){
                        nStepReturn += rewards[t + i] * gammaPow;
                        gammaPow *= Gamma;
                    }

                    added += lambdaPowed * nStepReturn;
                    lambdaPowed *= lambda;
                }

                returns[t] += (1 - lambda) * added;
            }

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
        