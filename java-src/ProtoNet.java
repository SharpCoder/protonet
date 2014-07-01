package protonet;

import java.util.ArrayList;
import java.util.Random;

public class ProtoNet {
	private Random rand;
	private Neuron[] inputs;
	private ArrayList<Neuron> flattened = null;

	public ProtoNet(int[] topology)
	{
		// Validate the input.
		if (topology == null || topology.length < 1) return;

		// NOTE: I seed my random for consistency when
		// testing the network.
		rand = new Random(1337);

		// And now I'm cheating with a flattened 
		// list of nodes. This makes training easier.
		flattened = new ArrayList<Neuron>();

		// Create the neuron array.
		inputs = new Neuron[topology[0]];

		// Generate the input nodes and recursively construct the graph.
		for (int i = 0; i < topology[0]; i++)
		{
			// The first layer of the topology arg will be
			// representative of the inputs.
			inputs[i] = new Neuron(topology[0], rand);
			flattened.add(inputs[i]);
		}

		// Recursively construct the graph.
		recurse(inputs, topology, 1);
	}

	public float GetValue(float[] input)
	{
		float result = 0f;

		// Iterate over each INPUT NODE.
		for (int i = 0; i < inputs.length; i++)
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
		for (int i = 0; i < inputs.length; i++)
		{
			inputs[i].CalculateError(target);
		}

		// Now train everything.
		for (int i = 0; i < flattened.size(); i++)
		{
			flattened.get(i).Train();
		}
	}

	private void recurse(Neuron[] parents, int[] topology, int index)
	{
		if (parents == null || parents.length == 0) return;
		if (topology == null || topology.length == 0) return;
		if (index >= topology.length) return;

		// Construct the next layer.
		Neuron[] layer = new Neuron[topology[index]];
		for (int i = 0; i < layer.length; i++)
		{
			// Create each node in this layer.
			layer[i] = new Neuron(parents.length, rand);
			flattened.add(layer[i]);
		}

		// "Wire up" the parents.
		for (int i = 0; i < parents.length; i++)
		{
			parents[i].Children = layer;
		}

		// Recurse.
		recurse(layer, topology, ++index);
	}
}
