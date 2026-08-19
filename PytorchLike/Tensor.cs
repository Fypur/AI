using System.Numerics.Tensors;

namespace AI
{
    public class Tensor
    {
        public float[] Data;
        public float[] Grad;
        public int[] Shape;
        public int[] Strides;

        public TensorOpp CreatedFrom;

        public Tensor(int[] shape)
        {
            if (shape == null)
                throw new Exception("Shape cannot be null");

            Shape = shape;

            Strides = new int[shape.Length];

            int size = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] <= 0)
                    throw new Exception("Shape array cannot contain 0s or negatives");
                size *= shape[i];
            }

            Strides[Strides.Length - 1] = 1;
            for (int i = Strides.Length - 2; i >= 0; i--)
                Strides[i] = Strides[i + 1] * shape[i + 1];

            Data = new float[size];
            Grad = new float[size];

            CreatedFrom = new TensorOppNoOpp();
        }

        public void Backward()
        {
            for (int i = 0; i < Grad.Length; i++)
                Grad[i] = 1f;

            CreatedFrom.Backward();
        }

        public static Tensor MatrixMult(Tensor a, Tensor b)
        {
            //determining shape and checking if we can matrix mult a and b
            if (a.Shape.Length != b.Shape.Length || a.Shape.Length < 2)
                throw new Exception("Shape lengths don't correspond for matrix multiplication");

            int[] newShape = new int[a.Shape.Length];
            int numMatrixMult = 1;

            for (int i = 0; i < a.Shape.Length - 2; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                    throw new Exception("Matrix multiplied Tensors don't have the same number of batches");

                newShape[i] = a.Shape[i];
                numMatrixMult *= a.Shape[i];
            }

            if (a.Shape[a.Shape.Length - 1] != b.Shape[b.Shape.Length - 2])
                throw new Exception("Incompatible matrix dimensions for multiplication");

            newShape[a.Shape.Length - 2] = a.Shape[a.Shape.Length - 2];
            newShape[a.Shape.Length - 1] = b.Shape[b.Shape.Length - 1];

            Tensor t = new Tensor(newShape);

            int aRows = a.Shape[a.Shape.Length - 2];
            int aCols = a.Shape[a.Shape.Length - 1];
            int bCols = b.Shape[b.Shape.Length - 1];

            int aMatrixSize = aRows * aCols;
            int bMatrixSize = aCols * bCols;
            int tMatrixSize = aRows * bCols;

            //Actual matrix mult code
            for (int i = 0; i < numMatrixMult; i++)
            {
                int aOffset = i * aMatrixSize;
                int bOffset = i * bMatrixSize;
                int tOffset = i * tMatrixSize;

                for (int r = 0; r < aRows; r++)
                {
                    for (int c = 0; c < aCols; c++)
                    {
                        float a_value = a.Data[aOffset + r * aCols + c];
                        Span<float> tRow = new Span<float>(t.Data, tOffset + r * t.Strides[t.Strides.Length - 2], bCols);
                        ReadOnlySpan<float> bRow = new ReadOnlySpan<float>(b.Data, bOffset + c * b.Strides[b.Strides.Length - 2], bCols);

                        TensorPrimitives.MultiplyAdd(bRow, a_value, tRow, tRow);
                    }
                }
            }

            return t;
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            if (a.Shape.Length != b.Shape.Length)
                throw new Exception("Tensors must have the same shape for addition.");

            for (int i = 0; i < a.Shape.Length; i++)
                if (a.Shape[i] != b.Shape[i])
                    throw new Exception("Tensors must have the same shape for addition.");

            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Add(a.Data, b.Data, t.Data);

            t.CreatedFrom = new TensorOppAdd(a, b, t);

            return t;
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            if (a.Shape.Length != b.Shape.Length)
                throw new Exception("Tensors must have the same shape for substraction.");

            for (int i = 0; i < a.Shape.Length; i++)
                if (a.Shape[i] != b.Shape[i])
                    throw new Exception("Tensors must have the same shape for substraction.");

            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Subtract(a.Data, b.Data, t.Data);

            t.CreatedFrom = new TensorOppSub(a, b, t);

            return t;
        }

        public static Tensor operator -(Tensor a)
        {
            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Negate(a.Data, t.Data);
            t.CreatedFrom = new TensorOppOpposite(a, t);

            return t;
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            if (a.Shape.Length != b.Shape.Length)
                throw new Exception("Tensors must have the same shape for multiplication.");

            for (int i = 0; i < a.Shape.Length; i++)
                if (a.Shape[i] != b.Shape[i])
                    throw new Exception("Tensors must have the same shape for multiplication.");

            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Multiply(a.Data, b.Data, t.Data);

            t.CreatedFrom = new TensorOppMult(a, b, t);

            return t;
        }

        public static Tensor operator *(Tensor a, float b)
        {
            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Multiply(a.Data, b, t.Data);

            t.CreatedFrom = new TensorOppMult(a, b, t);

            return t;
        }

        public static Tensor operator *(float b, Tensor a)
        {
            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Multiply(a.Data, b, t.Data);

            t.CreatedFrom = new TensorOppMult(a, b, t);

            return t;
        }

        public static Tensor operator /(Tensor a, float b)
        {
            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Multiply(a.Data, 1 / b, t.Data);

            t.CreatedFrom = new TensorOppMult(a, 1 / b, t);

            return t;
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            if (a.Shape.Length != b.Shape.Length)
                throw new Exception("Tensors must have the same shape for multiplication.");

            for (int i = 0; i < a.Shape.Length; i++)
                if (a.Shape[i] != b.Shape[i])
                    throw new Exception("Tensors must have the same shape for multiplication.");

            Tensor t = new Tensor(a.Shape);

            t.Data = new float[a.Data.Length];
            TensorPrimitives.Divide(a.Data, b.Data, t.Data);


            t.CreatedFrom = new TensorOppDivide(a, b, t);

            return t;
        }
    }
}
