// Author: SharpCoder
// Date: 2013-10-09
// Notes: This is my basic neural network
// framework implementing the backpropagation
// algorithm using the sigmoid activation function.

using System;
using System.Collections.Generic;
namespace ProtoNetwork
{
	class ProtoNet
	{
		private Random rand;
		private Neuron[] inputs;
		private List<Neuron> flattened;

		public ProtoNet(int[] topology)
		{
			// Validate the input.
			if (topology == null || topology.Length < 1) return;

			// NOTE: I seed my random for consistency when
			// testing the network.
			rand = new Random(1337);
			
			// And now I'm cheating with a flattened 
			// list of nodes. This makes training easier.
			flattened = new List<Neuron>();

			// Create the neuron array.
			inputs = new Neuron[topology[0]];

			// Generate the input nodes and recursively construct the graph.
			for (var i = 0; i < topology[0]; i++)
			{
				// The first layer of the topology arg will be
				// representative of the inputs.
				inputs[i] = new Neuron(topology[0], rand);
				flattened.Add(inputs[i]);
			}

			// Recursively construct the graph.
			recurse(inputs, topology, 1);
		}

		public float GetValue(float[] input)
		{
			float result = 0f;

			// Iterate over each INPUT NODE.
			for (var i = 0; i < inputs.Length; i++)
			{
				// Feed each input node ALL of the inputs
				// sent into this function (very important).
				result += inputs[i].Evaluate(input);
			}

			// Return all the results that we got back.
			return result;
		}

		public void Train(float[] input, float target)
		{
			// This function is similar to GetValue()
			// except it uses the backpropagation algorithm
			// to train the network.
			float output = GetValue(input);
			
			// Calculate the error matrix with current output values
			// and before we change any weights.
			for (var i = 0; i < inputs.Length; i++)
			{
				inputs[i].CalculateError(target);
			}

			// Now train everything.
			for (var i = 0; i < flattened.Count; i++)
			{
				flattened[i].Train();
			}
		}

		private void recurse(Neuron[] parents, int[] topology, int index)
		{
			if (parents == null || parents.Length == 0) return;
			if (topology == null || topology.Length == 0) return;
			if (index >= topology.Length) return;

			// Construct the next layer.
			Neuron[] layer = new Neuron[topology[index]];
			for (var i = 0; i < layer.Length; i++)
			{
				// Create each node in this layer.
				layer[i] = new Neuron(parents.Length, rand);
				flattened.Add(layer[i]);
			}

			// "Wire up" the parents.
			for (var i = 0; i < parents.Length; i++)
			{
				parents[i].Children = layer;
			}

			// Recurse.
			recurse(layer, topology, ++index);
		}
	}

	class Neuron
	{
		const bool USE_BIAS = true;
		const float LEARN_RATE = 0.5f;

		public Neuron[] Children;

		private Random rand;
		private float error;
		private float output;
		private float[] weights;
		private float[] inputs;
		private int index = 0;

		public Neuron(int inputCount, Random rnd = null)
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
				inputs[inputs.Length - 1] = 1.0f;
			}
			else
			{
				inputs[inputs.Length - 1] = 0.0f;
			}

			// Now generate the random weights with
			// values between -1 and 1.
			for (var i = 0; i < weights.Length; i++)
			{
				weights[i] = (float)(rand.NextDouble() - rand.NextDouble());
			}
		}

		public float Evaluate(float[] input)
		{
			float result = 0f;

			// Set the inputs.
			for (var i = 0; i < input.Length; i++)
			{
				setValue(input[i]);
			}

			// Check if the neuron has "fired" (ie: all inputs are full).
			if (index != 0) return 0f;

			// Now calculate our value and propogate outwards
			this.output = GetValue();

			// Iterate over the children.
			if (Children != null && Children.Length > 0)
			{
				// If we are not an output node...
				for (var i = 0; i < Children.Length; i++)
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
			var result = 0f;
			// Iterate over each input
			for (var i = 0; i < inputs.Length; i++)
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
			for (var i = 0; i < weights.Length; i++)
			{
				weights[i] += LEARN_RATE * inputs[i] * error;
			}
		}

		public float CalculateError(float target)
		{
			if (this.error != 0) return this.error;

			// Calculate the error.
			if (Children == null || Children.Length == 0)
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
				for (var i = 0; i < Children.Length; i++)
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
			if (Children != null && Children.Length > index)
			{
				for (var r = 0; r < Children[index].weights.Length; r++)
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
			if (index > weights.Length - 2) index = 0;
		}
	}
}
