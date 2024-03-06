using System;
using Fiourp;

namespace AI
{
    public class DeepQAgent2
    {
        public NN2 Network;
        public NN2 TargetNetwork;

        public int StateSize {get; private set;}
        public float Epsilon = 1;
        public PrioritizedExperienceReplay2 replayBuffer;
        

        public DeepQAgent2(int[] layers, float learningRate = 0.001f, float movingAverageBeta = 0.9f,
         int memorySize = 65536, float replayAlpha = 0.6f, float replayBeta = 0.4f){

            replayBuffer = new PrioritizedExperienceReplay2(memorySize, replayAlpha, replayBeta);
            Network = new NN2(layers, learningRate, movingAverageBeta);
            TargetNetwork = Network.Copy();

        }

        public int Act(){
            float r = Rand.NextDouble();
            if(r < Epsilon)

        }

        public void Replay(){
            
        }
    }
}
