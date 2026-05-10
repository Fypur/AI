namespace AI.DeepQ
{
    public class Experience
    {
        public float[] State { get; set; }
        public int Action { get; set; }
        public float Reward { get; set; }
        public float[] NextState { get; set; }
        public bool Done { get; set; }
        public Experience(float[] state, int action, float reward, float[] nextState, bool done)
        {
            State = state;
            Action = action;
            Reward = reward;
            NextState = nextState;
            Done = done;
        }
    }
}
