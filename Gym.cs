using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI
{
    public abstract class Gym
    {
        public DeepQAgent2 Agent;

        public uint Step;
        public int EpisodeStep;
        public int GraphRefreshRate;
        public int LossGraphClearRate;

        private ScottPlot.Plot rewardGraph;
        private ScottPlot.Plottables.DataLogger dataLogger;
        private float totalReward;

        protected abstract string rewardGraphSaveLocation { get; }

        public Gym(DeepQAgent2 agent, int lossGraphRefreshRate = 3000, int lossGraphClearRate = 21000)
        {
            this.Agent = agent;
            GraphRefreshRate = lossGraphRefreshRate;
            LossGraphClearRate = lossGraphClearRate;

            rewardGraph = new ScottPlot.Plot();
            dataLogger = rewardGraph.Add.DataLogger();
        }

        public void DoStep()
        {
            if(Agent.TrainingStarted)
                Step++;

            EpisodeStep++;

            float[] state = GetState();

            int action = Agent.Act(state);

            float reward = UpdateAndReward(action);

            float[] nextState = GetState();
            bool done = Done();

            

            Agent.Remember(new Experience(state, action, reward, nextState, done));
            Agent.Replay();

            totalReward += reward;
            if (done)
            {
                dataLogger.Add(totalReward);
                Reset();
                EpisodeStep = 0;
                totalReward = 0;
            }

            if (Agent.TrainingStarted && Step % GraphRefreshRate == 0)
            {
                Agent.ExportLossGraph();
                ExportRewardGraph();
            }

            if (Agent.TrainingStarted && Step % LossGraphClearRate == 0)
                Agent.ClearLossGraph();
        }

        private void ExportRewardGraph()
        {
            rewardGraph.SavePng(rewardGraphSaveLocation, 1000, 1000);
            Console.WriteLine("Reward Graph exported step: " + Step);
        }

        protected abstract float UpdateAndReward(int action);
        protected abstract bool Done();
        protected abstract float[] GetState();
        protected abstract void Reset();
        public virtual void Render() { }
    }
}
