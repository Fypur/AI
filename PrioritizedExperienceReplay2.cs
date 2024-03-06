namespace AI
{
    public class PrioritizedExperienceReplay2
    {
        public int Capacity { get; private set; }
        public float Alpha;
        public float Beta;
        public float BetaIncrease;
        public float Epsilon;
        public int BatchSize;
        public bool Filled;

        private SumTree2 sumTree;
        private MinTree minTree;
        private float MaxPriority = 1;
        private Experience[] data;
        private int iMemory;

        public PrioritizedExperienceReplay2(int capacity, float alpha, float beta, float betaIncrease, float epsilon, int batchSize)
        {
            int pow2 = 1;
            while (pow2 < capacity)
                pow2 *= 2;

            Capacity = pow2;
            Alpha = alpha;
            Beta = beta;
            BetaIncrease = betaIncrease;
            Epsilon = epsilon;
            BatchSize = batchSize;

            if ((double)1 / Capacity < epsilon)
                throw new Exception("Base probability epsilon is too high");

            MaxPriority = 1;

            data = new Experience[capacity];
            sumTree = new SumTree2(capacity);
            minTree = new MinTree(capacity);
        }

        public void Add(Experience experience)
        {
            data[iMemory] = experience;

            float alphaPriority = (float)Math.Pow(MaxPriority, Alpha);

            sumTree.ChangeWeight(iMemory, alphaPriority);
            minTree.ChangeWeight(iMemory, alphaPriority);
            
            iMemory++;
            if(iMemory >= data.Length)
            {
                iMemory = 0;
                Filled = true;
            }
        }

        public void ChangePriority(int index, float priority)
        {
            if (!float.IsNormal(priority + Epsilon))
                throw new Exception("Priority is not normal");

            MaxPriority = Math.Max(MaxPriority, priority + Epsilon);

            float prob = (float)Math.Pow(priority + Epsilon, Alpha);
            sumTree.ChangeWeight(index, prob);
            minTree.ChangeWeight(index, prob);
        }

        public Sample Sample()
        {
            Sample sample = new();
            sample.Experience = new Experience[BatchSize];
            sample.Weights = new float[BatchSize];
            sample.Indices = new int[BatchSize];

            Beta = Math.Min(1, Beta + BetaIncrease);

            double maxWeight = Math.Pow(Capacity * minTree.Minimum() / sumTree.Sum(), -Beta);

            for(int i = 0; i < BatchSize; i++)
            {
                sample.Indices[i] = sumTree.Sample();
                sample.Experience[i] = data[sample.Indices[i]];
                sample.Weights[i] = (float)(Math.Pow(Capacity * sumTree.GetWeight(sample.Indices[i]) / sumTree.Sum(), -Beta) / maxWeight);
            }

            return sample;
        }

        public void Save(string path)
        {
            if (!Filled)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Saving replay buffer before it's filled");
                Console.ForegroundColor = ConsoleColor.White;
                return;
            }

            if (!path.EndsWith(".json")) path += ".json";

            Tuple<Experience[], float[], int, float> saved = new Tuple<Experience[], float[], int, float>(new Experience[sumTree.Capacity], new float[sumTree.Capacity], iMemory, MaxPriority);
            for (int i = 0; i < sumTree.Capacity; i++)
            {
                saved.Item1[i] = data[i];
                saved.Item2[i] = sumTree.GetWeight(i);
            }

            System.IO.File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(saved));
        }

        public void Load(string path)
        {
            if (!path.EndsWith(".json")) path += ".json";

            Tuple<Experience[], float[], int, float> saved = System.Text.Json.JsonSerializer.Deserialize<Tuple<Experience[], float[], int, float>>(System.IO.File.ReadAllText(path));

            for(int i = 0; i < saved.Item1.Length; i++)
            {
                data[i] = saved.Item1[i];
                sumTree.ChangeWeight(i, saved.Item2[i]);
                minTree.ChangeWeight(i, saved.Item2[i]);
            }

            iMemory = saved.Item3;
            MaxPriority = saved.Item4;
            Filled = true;
        }
    }

    public class Sample
    {
        public Experience[] Experience;
        public float[] Weights;
        public int[] Indices;
    }
}
