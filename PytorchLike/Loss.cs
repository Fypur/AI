namespace AI
{
    public abstract class Loss
    {
        public abstract Tensor CalcLoss(Tensor predictions, Tensor targets);
    }
}
