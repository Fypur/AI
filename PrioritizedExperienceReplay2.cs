namespace AI
{
    public class PrioritizedExperienceReplay2
    {
        public int Capacity { get; private set; }
        public float Alpha;
        public float Beta;
        public int BatchSize;

        private SumTree2 sumTree;
        private MinTree minTree;
        private float MaxPriority;
        private Experience[] data;
        private int iMemory;

        public PrioritizedExperienceReplay2(int capacity, float alpha, float beta, int batchSize){
            Capacity = capacity;
            Alpha = alpha;
            Beta = beta;
            BatchSize = batchSize:
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
                iMemory = 0;
        }

        public void ChangeProbability(int index, float priority)
        {
            MaxPriority = Math.Max(MaxPriority, priority);
            float prob = (float)Math.Pow(priority, Alpha);
            sumTree.ChangeWeight(index, prob);
            minTree.ChangeWeight(index, prob);
        }

        public Sample Sample()
        {
            Sample sample = new();
            sample.Experience = new Experience[BatchSize];
            sample.Weights = new float[BatchSize];
            sample.Indices = new int[BatchSize];

            for(int i = 0; i < Capacity; i++){
                sample.Indices[i] = sumTree.Sample();
                sample.Experience[i] = data[sample.Indices[i]];
                sample.Weights[i] = ;
            }

            return sample;
        }
    }

    public class Sample{
        public Experience[] Experience;
        public float[] Weights;
        public int[] Indices;
    }
}
