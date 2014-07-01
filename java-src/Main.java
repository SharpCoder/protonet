package protonet;


public class Main {

	static int epoch = 100000;
	public static void main (String args[]){
		System.out.println("Hello, world!");
		System.out.println("Initializing protonetwork...");
		
		ProtoNet network = new ProtoNet(new int[] { 2, 2, 1 });
		
		// Now we iterate over our epoch count.
		for (int i = 0; i < epoch; i++)
		{
			// And train the network with the desired information
			network.Train(new float[] { 0, 0 }, 0);
			network.Train(new float[] { 1, 0 }, 1);
			network.Train(new float[] { 0, 1 }, 1);
			network.Train(new float[] { 1, 1 }, 0);
		}

		// Write our output.
		System.out.println("Proto Network trained @" + epoch + " epochs.");
		System.out.println("<0,0> = " + network.GetValue(new float[] { 0, 0 }));
		System.out.println("<1,0> = " + network.GetValue(new float[] { 1, 0 }));
		System.out.println("<0,1> = " + network.GetValue(new float[] { 0, 1 }));
		System.out.println("<1,1> = " + network.GetValue(new float[] { 1, 1 }));
		
		// Terminate.
		System.out.println("Terminating...");
	}
}
