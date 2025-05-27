using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AI
{
    public class Rand
    {
        public static Random random = new Random();

        /// <summary>
        /// Random integer within specified range, where maxValue is excluded
        /// </summary>
        /// <param name="minValue">Included</param>
        /// <param name="maxValue">Excluded</param>
        /// <returns></returns>
        public static int NextInt(int minValue, int maxValue)
            => random.Next(minValue, maxValue);

        public static float NextFloat(float min, float max)
            => (float)(random.NextDouble() * (max - min) + min);

        public static float NextDouble()
            => (float)random.NextDouble();
    }
}
