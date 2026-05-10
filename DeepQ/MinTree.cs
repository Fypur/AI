using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI
{
    public class MinTree
    {
        public int Capacity; //leaf amount
        private float[] weights;

        public MinTree(int capacity)
        {
            Capacity = capacity;
            int nodeAmount = Capacity * 2 - 1;

            weights = new float[nodeAmount];
            for(int i = 0; i < weights.Length; i++)
                weights[i] = float.PositiveInfinity;
        }

        public MinTree(float[] data)
        {
            Capacity = data.Length;
            int nodeAmount = data.Length * 2 - 1;

            weights = new float[nodeAmount];
            for (int i = 0; i < weights.Length; i++)
                weights[i] = float.PositiveInfinity;
            for (int i = 0; i < data.Length; i++)
                ChangeWeight(i, data[i]);
        }

        public void ChangeWeight(int index, float newWeight)
        {
            index = GetLeafIndex(index);
            weights[index] = newWeight;

            Backprop(index - (index + 1) % 2);
        }

        private int GetLeafIndex(int index)
            => index + Capacity - 1; //i have no idea why tf this works this is black magic

        private void Backprop(int index)
        {
            while (index > 0)
            {
                float min = (float)Math.Min(weights[index], weights[index + 1]);
                index = (index - 1) / 2; //parent index
                weights[index] = min;
            }
        }

        public float Minimum()
            => weights[0];
    }
}
