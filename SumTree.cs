using System;

namespace AI
{
    public class SumTree<T>
    {
        private static Random random = new();
        public int Capacity; //leaf amount
        private T[] data;
        private float[] weights;
        private static Mutex mRand = new Mutex();

        public SumTree(T[] data, float[] dataWeights)
        {
            if (data.Length != dataWeights.Length) throw new Exception("data length and weights length mismatch");

            this.data = data;
            Capacity = data.Length;
            int nodeAmount = dataWeights.Length * 2 - 1;

            weights = new float[nodeAmount];
            for (int i = 0; i < dataWeights.Length; i++)
                ChangeWeight(i, dataWeights[i]);
        }

        public void ChangeValue(int index, T newValue, float newWeight)
        {
            data[index] = newValue;

            index = GetLeafIndex(index);
            float diff = newWeight - weights[index];
            weights[index] = newWeight;

            BackpropDiff(index, diff);
        }

        public void ChangeWeight(int index, float newWeight)
        {
            index = GetLeafIndex(index);
            float diff = newWeight - weights[index];
            weights[index] = newWeight;

            BackpropDiff(index, diff);
        }

        public T GetValue(int index)
            => data[index];

        public float GetWeight(int index)
            => weights[GetLeafIndex(index)];

        public float Sum()
            => weights[0];

        private int GetLeafIndex(int index)
            => index + Capacity - 1; //i have no idea why tf this works this is black magic

        private void BackpropDiff(int index, float diff)
        {
            while (index > 0)
            {
                index = (index - 1) / 2; //parent index
                weights[index] += diff;
            }
        }

        public Tuple<T, int> Sample()
        {
            int index = 0;

            mRand.WaitOne();
            double rand = random.NextDouble() * weights[0]; //Random is not thread safe
            mRand.ReleaseMutex();

            while(2 * index + 1 < weights.Length) //child is still in the tree
            {
                //left child weight
                if (rand < weights[index * 2 + 1])
                    index = index * 2 + 1; //now on left index
                else
                {
                    rand -= weights[index * 2 + 1]; //minus left weight
                    index = index * 2 + 2; //now on right index
                }
            }

            int outIndex = index - Capacity + 1;
            return new(data[outIndex], outIndex);
        }
    }
}
