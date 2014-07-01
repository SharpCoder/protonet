package protonet;

public class MathHelper {
	public static float Sigmoid(float input)
	{
		return (float)(1f/(1f + Math.exp(-input)));
	}
}
