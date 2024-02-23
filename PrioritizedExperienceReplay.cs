using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json.Serialization;

namespace AI
{
    public class PrioritizedExperienceReplay
    {
        private SumTree<Experience> sumTree;
        private MinTree minTree;
        private Tuple<Experience, float>[] initMemory;
        private int iMemory;
        public bool Filled;
        public float MaxPriority;

        public PrioritizedExperienceReplay(int capacity)
        {
            initMemory = new Tuple<Experience, float>[capacity];
            MaxPriority = 1;
        }

        public PrioritizedExperienceReplay(int iMemory, Tuple<Experience, float>[] initMemory, SumTree<Experience> sumTree, MinTree minTree)
        {
            this.iMemory = iMemory;
            this.initMemory = initMemory;
            this.sumTree = sumTree;
            this.minTree = minTree;
        }

        public void Add(Experience experience, float probability)
        {
            MaxPriority = Math.Max(MaxPriority, probability);

            if (!Filled)
            {
                initMemory[iMemory] = new(experience, probability);
                iMemory++;

                if (initMemory.Length <= iMemory)
                {
                    iMemory = 0;
                    Filled = true;
                    InitializeTree();
                }

                return;
            }

            sumTree.ChangeValue(iMemory, experience, probability);
            minTree.ChangeWeight(iMemory, probability);
            iMemory++;
            if(iMemory >= sumTree.Capacity)
                iMemory = 0;
        }

        public void UpdatePriority(int index, float newProbability)
        {
            MaxPriority = Math.Max(MaxPriority, newProbability);
            sumTree.ChangeWeight(index, newProbability);
            minTree.ChangeWeight(index, newProbability);
        }

        public Experience GetValue(int index)
            => sumTree.GetValue(index);

        public float GetPriority(int index)
            => sumTree.GetWeight(index);

        public void UpdateMaxPriority()
        {
            MaxPriority = sumTree.GetWeight(0);
            for(int i = 1; i < sumTree.Capacity; i++)
                MaxPriority = Math.Max(MaxPriority, sumTree.GetWeight(i));
        }

        private void InitializeTree()
        {
            Check();
            Experience[] exps = new Experience[initMemory.Length];
            float[] weights = new float[initMemory.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                exps[i] = initMemory[i].Item1;
                weights[i] = initMemory[i].Item2;
            }

            sumTree = new SumTree<Experience>(exps, weights);
            minTree = new MinTree(weights);
            initMemory = null;
        }

        public Tuple<Experience, int>[] Sample(int batchSize)
        {
            if (!Filled) throw new Exception("Cannot batch before memory is filled");

            Tuple<Experience, int>[] minibatch = new Tuple<Experience, int>[batchSize];
            for (int i = 0; i < minibatch.Length; i++)
                minibatch[i] = sumTree.Sample();

            return minibatch;
        }

        public float MinProbability()
            => minTree.Minimum();

        public float ProbabilitySum()
            => sumTree.Sum();

        public void Save(string path)
        {
            if (initMemory != null) throw new Exception("Can't save replay buffer before it is filled");

            if (!path.EndsWith(".json")) path += ".json";

            Tuple<Experience[], float[], int, float> data = new Tuple<Experience[], float[], int, float>(new Experience[sumTree.Capacity], new float[sumTree.Capacity], iMemory, MaxPriority);
            for (int i = 0; i < sumTree.Capacity; i++)
            {
                data.Item1[i] = sumTree.GetValue(i);
                data.Item2[i] = sumTree.GetWeight(i);
            }

            System.IO.File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(data));
        }

        public void Load(string path)
        {
            if (!path.EndsWith(".json")) path += ".json";

            Tuple<Experience[], float[], int, float> data = System.Text.Json.JsonSerializer.Deserialize<Tuple<Experience[], float[], int, float>>(System.IO.File.ReadAllText(path));
            sumTree = new(data.Item1, data.Item2);
            minTree = new(data.Item2);
            iMemory = data.Item3;
            MaxPriority = data.Item4;
            initMemory = null;
            Filled = true;
        }

        public void Check()
        {
            void Throw() => throw new Exception("Memory has not normal float values");

            for (int i = 0; i < initMemory.Length; i++)
            {
                Experience check = initMemory[i].Item1;
                for (int j = 0; j < check.State.Length; j++)
                {
                    if (!float.IsNormal(check.State[j]) && check.State[j] != 0)
                        Throw();
                    if (!float.IsNormal(check.NextState[j]) && check.NextState[j] != 0)
                        Throw();
                }

                if (!float.IsNormal(check.Reward) && check.Reward != 0)
                    Throw();
            }
        }
    }
}
