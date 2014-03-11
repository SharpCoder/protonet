// Author: SharpCoder
// Date: 2014-03-10
// Notes: This is a basic back propagation
// neural network with a hard-coded topology
// of 2-2-1 which demonstrates the lightest-weight
// approach possible and is going to be part of a larger,
// more adaptable neural network framework.
//
// NOTE: This design does not use a bias. If you run
// some tests, you will notice it takes considerably longer
// to hone in on the proper weights that can represent the XOR
// challenge.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProtoNetwork
{
	public class FastNet
	{
		// This is the learn rate. It is suggested you start at 0.5 but tinker with it
		// based on your needs.
		const float LEARN_RATE = 0.5f;

		// These are the necessary variables for our neural network. Not much is really
		// required.
		Random rand;
		float[] weights;
		float[] outputs;

		public FastNet()
		{
			// I seed the random to a specific value for consistency
			// through multiple iterations of my program.
			this.rand = new Random(424242);

			// Since we have a topology of 2-2-1, we know there are 10 weights.
			// Check out my website for more information. 
			// http://debuggle.com/Home/ProtoNet
				
			weights = new float[10];
			outputs = new float[5];

			// Initialize the weights.
			for (int i = 0; i < weights.Length; i++)
				// This will generate a number between -1f and 1f (exclusive).
				weights[i] = (float)(rand.NextDouble() - rand.NextDouble());
		}

		public float getOutput(float[] input)
		{
			// Now we get the output of the neural network. This function is relatively simple
			// though a doozey to look at. Basically, we are just taking the input of a given node
			// and multiplying it by the associated weight.
			outputs[0] = sig(input[0] * weights[0] + input[1] * weights[1]);
			outputs[1] = sig(input[0] * weights[2] + input[1] * weights[3]);
			outputs[2] = sig(outputs[0] * weights[4] + outputs[1] * weights[5]);
			outputs[3] = sig(outputs[0] * weights[6] + outputs[1] * weights[7]);
			outputs[4] = sig(outputs[2] * weights[8] + outputs[3] * weights[9]);

			// The output of the final node is the output of the whole graph
			// since there is only 1 output node.
			return outputs[4];
		}

		public void train(float[] input, float target)
		{
			// Now the fun part, training the network!
			// This operation includes calculating the derivative of our sigmoid output. To be specific, we
			// are taking the derivative of each node's mathematical operation, with respect to the 
			// weights and nodes that utilized it's output.
			//
			// Each node is "blamed" equal to the raw 'distance' of it's output relative to the target
			// times the error of each node that utilizes it's value. So the closer to the "start" of the graph
			// the less blame you will get.

			float[] errors = new float[outputs.Length];
			float output = getOutput(input);

			// Compute the error matrix. This operation is really just the derivative of the nodes output
			// times the error of the nodes below it.
			errors[4] = output * (1f - output) * (target - output);
			errors[3] = outputs[3] * (1f - outputs[3]) * (errors[4] * weights[9]);
			errors[2] = outputs[2] * (1f - outputs[2]) * (errors[4] * weights[8]);
			errors[1] = outputs[1] * (1f - outputs[1]) * ((errors[3] * weights[7]) + (errors[2] * weights[5]));
			errors[0] = outputs[0] * (1f - outputs[0]) * ((errors[3] * weights[6]) + (errors[2] * weights[4]));

			// Now we compute the adjustment. This is straightforward, each weight gets adjusted
			// equal to the LEARN_RATE times the value that weight was multiplied against times
			// the error of the node which "houses" the weight.
			weights[9] += LEARN_RATE * outputs[3] * errors[4];
			weights[8] += LEARN_RATE * outputs[2] * errors[4];
			weights[7] += LEARN_RATE * outputs[1] * errors[3];
			weights[6] += LEARN_RATE * outputs[0] * errors[3];
			weights[5] += LEARN_RATE * outputs[1] * errors[2];
			weights[4] += LEARN_RATE * outputs[0] * errors[2];
			weights[3] += LEARN_RATE * input[1] * errors[1];
			weights[2] += LEARN_RATE * input[0] * errors[1];
			weights[1] += LEARN_RATE * input[1] * errors[0];
			weights[0] += LEARN_RATE * input[0] * errors[0];

			// And that's it!
		}

		private float sig(float val)
		{
			// This is the sigmoid activation function. Essentially it will compress any number into a value between 0 and 1.
			return (float)(1.0f / (1.0f + Math.Exp(-val)));
		}

	}
}
