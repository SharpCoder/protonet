﻿// Author: SharpCoder
// Date: 2013-10-09
// Notes: This is an exmaple file for using
// my neural network framework.

using System;
namespace ProtoNetwork
{
	class Program
	{
		const int epoch = 2000;
		static void Main(string[] args)
		{
			// Create our new network with the following topology.
			//
			//   ---     ---
			//  |   |   |   |
			//   ---     ---
			//    \      /
			//      \  /
			//      ---
			//     |   |
			//      ---
			//
			PrintHeader();
			ProtoNet network = new ProtoNet(new int[] { 2, 1 });
		
			// Now we iterate over our epoch count.
			for (var i = 0; i < epoch; i++)
			{
				// And train the network with the desired information
				network.Train(new float[] { 0, 0 }, 0);
				network.Train(new float[] { 1, 0 }, 1);
				network.Train(new float[] { 0, 1 }, 1);
				network.Train(new float[] { 1, 1 }, 0);
			}

			// Write our output.
			Console.WriteLine("<0,0> = " + network.GetValue(new float[] { 0, 0 }));
			Console.WriteLine("<1,0> = " + network.GetValue(new float[] { 1, 0 }));
			Console.WriteLine("<0,1> = " + network.GetValue(new float[] { 0, 1 }));
			Console.WriteLine("<1,1> = " + network.GetValue(new float[] { 1, 1 }));

			Console.WriteLine("");
			Console.WriteLine("Finished with '" + epoch + "' epochs.");
			Console.ReadKey();
		}

		static void PrintHeader()
		{
			Console.WriteLine("Proto-Net v0.0.1");
			Console.WriteLine("Created by: SharpCoder");
			Console.WriteLine("");
			Console.WriteLine("This simple project demonstrates a neural network using");
			Console.WriteLine("as simplistic and well documented code as I could. I hope");
			Console.WriteLine("to refine the implementation over time, offering a more cohesive");
			Console.WriteLine("solution. But for now, I hope this project helps further elemantary");
			Console.WriteLine("understanding of the backpropagation algorithm.");
			Console.WriteLine("\n");
		}
	}
}
