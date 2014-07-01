package protonet;

import java.util.Random;

public class Neuron
{
	boolean USE_BIAS = true;
	float LEARN_RATE = 0.5f;

	public Neuron[] Children;

	private Random rand;
	private float error;
	private float output;
	private float[] weights;
	private float[] inputs;
	private int index = 0;

	public Neuron(int inputCount, Random rnd)
	{
		// Specify the ability to pass a random object
		// for consistency. This will make sure the nodes
		// don't all have the same starting values (because if enough nodes
		// were generated at once, the randoms would all have the same seed potentially).
		if (rnd != null) rand = rnd;
		else rand = new Random();

		// Now wire up the weights and inputs.
		weights = new float[inputCount + 1];
		inputs = new float[inputCount + 1];

		// If we want to use a bias, set the last input to a 1.
		// Otherwise a zero to make it basically useless.
		if (USE_BIAS)
		{
			inputs[inputs.length - 1] = 1.0f;
		}
		else
		{
			inputs[inputs.length - 1] = 0.0f;
		}

		// Now generate the random weights with
		// values between -1 and 1.
		for (int i = 0; i < weights.length; i++)
		{
			weights[i] = (float)(rand.nextDouble() - rand.nextDouble());
		}
	}

	public float Evaluate(float[] input)
	{
		float result = 0f;

		// Set the inputs.
		for (int i = 0; i < input.length; i++)
		{
			setValue(input[i]);
		}

		// Check if the neuron has "fired" (ie: all inputs are full).
		if (index != 0) return 0f;

		// Now calculate our value and propogate outwards
		this.output = GetValue();

		// Iterate over the children.
		if (Children != null && Children.length > 0)
		{
			// If we are not an output node...
			for (int i = 0; i < Children.length; i++)
			{
				result += Children[i].Evaluate(new float[] { output });
			}
		}
		else
		{
			// Otherwise this is an output node, so return our value.
			result = output;
		}

		return result;
	}

	public float GetValue()
	{
		float result = 0f;
		// Iterate over each input
		for (int i = 0; i < inputs.length; i++)
		{
			// And calculate the value.
			result += inputs[i] * weights[i];
		}

		// Run our value through the activation function.
		this.output = MathHelper.Sigmoid(result);
		return output;
	}

	public void Train()
	{
		// Iterate over our weights and update them
		// based on the error calculations.
		for (int i = 0; i < weights.length; i++)
		{
			weights[i] += LEARN_RATE * inputs[i] * error;
		}
	}

	public float CalculateError(float target)
	{
		if (this.error != 0) return this.error;

		// Calculate the error.
		if (Children == null || Children.length == 0)
		{
			// This is the output node, so we do a slightly
			// different calculation.
			error = output * (1f - output) * (target - output);
		}
		else
		{
			// NOTE: We're basically taking the derivative of our
			// activation function and plugging in all the numbers. That's
			// why this math may look strange.
			error = output * (1f - output);
			float temp = 0f;

			// Iterate over each child and calculate the error
			// associated with the weight that connects to it.
			for (int i = 0; i < Children.length; i++)
			{
				// At it's simplest form, the error is equal to
				// the error of all the things below us times the weights that 
				// used our nodes output.
				temp += Children[i].CalculateError(target) * getChildWeight(i);
			}

			// So we calculate all those errors and multiply it to our own. Basically
			// the closer to the "top" of the graph you get, the less blamed 
			// those nodes will be..
			error *= temp;
		}

		// Return our error value.
		return error;
	}


	private float getChildWeight(int index)
	{
		// This method will look at the [index] child and
		// find the weight that was used in a calculation based
		// on this current nodes output.
		if (Children != null && Children.length > index)
		{
			for (int r = 0; r < Children[index].weights.length; r++)
			{
				if (Children[index].inputs[r] == this.output)
					return Children[index].weights[r];
			}
		}

		// If no match is found, return one.
		// NOTE: This should never happen unless it's an output node.
		// Which still shouldn't happen.
		return 1f;
	}

	private void setValue(float input)
	{
		this.error = 0f;
		// Set the value in our array.
		inputs[index++] = input;
		// If we are in the "bias" territory, reset the index.
		if (index > weights.length - 2) index = 0;
	}
}