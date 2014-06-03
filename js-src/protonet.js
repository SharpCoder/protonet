// Author: SharpCoder
// Date: 2014-06-03
// Notes: This is my basic neural network
// framework implementing the backpropagation
// algorithm using the sigmoid activation function.

MathHelper = (function() {
	return {
		Sigmoid: function( input ) {
			return ( 1.0 / (1.0 + Math.exp(-input)) );
		}
	};
})();

ProtoNet = function( topology ) {
	
	// Validate the input.
	if ( topology === undefined || topology.length < 1 ) return;
	
	// Instantiate the arrays and create the properties.
	var self = this;
	this.inputs = new Array();
	this.flattened = new Array();
	
	// Create the input neurons.
	for ( var i = 0; i < topology[0]; i++ ) {
		// The first layer of the topology argument will be
		// representative of the inputs.
		this.inputs.push( new Neuron( topology[0] ) );
		this.flattened.push( this.inputs[i] );
	}
	
	function recurse( parents, topology, index ) {
		
		if ( parents === undefined || parents.length == 0 ) return;
		if ( topology === undefined || topology.length == 0 ) return;
		if ( index >= topology.length ) return;
		
		// Construct the next layer.
		var layer = new Array();
		for ( var i = 0; i < topology[index]; i++ ) {
			// Create each node in this layer.
			layer.push( new Neuron( parents.length) );
			self.flattened.push( layer[i] );
		}
		
		// Wire up the parents.
		for ( var i = 0; i < parents.length; i++ ) {
			parents[i].Children = layer;
		}
		
		// Recursively iterate.
		recurse( layer, topology, ++index );
	}
	
	// Recursively construct the graph.
	recurse( this.inputs, topology, 1);		
};

ProtoNet.prototype.GetValue = function( input ) {
	var result = 0;
	
	// Iterate over each input node.
	for ( var i = 0; i < this.inputs.length; i++ ) {
		result += this.inputs[i].Evaluate( input );
	}
	
	// Return all the results that we got back.
	return result;
}

ProtoNet.prototype.Train = function( input, target ) {
	
	// This function is similar to GetValue()
	// except it uses the backpropagation algorithm
	// to train the network.
	var output = this.GetValue( input );
	
	// Calculate the error matrix with current output values
	// and before we change any weights.
	for ( var i = 0; i < this.inputs.length; i ++ ) {
		this.inputs[i].CalculateError( target );
	}
	
	// Now train everything.
	for ( var i = 0; i < this.flattened.length; i++ ) {
		this.flattened[i].Train();
	}
}

Neuron = function( inputCount ) {
	
	this.USE_BIAS = true;
	this.LEARN_RATE = 0.5;
	this.Children = new Array();
	
	// Private members.
	this.error = 0.0;
	this.output = 0.0;
	this.index = 0;
	this.inputLength = inputCount;
	this.weights = new Array();
	this.inputs = new Array();
	
	// Wire up the inputs and weights.
	for ( var i = 0; i < inputCount + 1; i++ ) {
		this.weights.push( Math.random() - Math.random() );
	}
	
	// Set the bias.
	if ( this.USE_BIAS ) this.weights[this.weights.length - 1] = 1.0;
	else this.weights[this.weights.length - 1] = 0.0;
	
};

Neuron.prototype.Evaluate = function( input ) {
	if ( input === undefined || input.length == 0 ) return;
	var result = 0;
	
	// Iterate over each item in the input array.
	for ( var i = 0; i < input.length; i++ ) {
		// Set the value.
		this.SetValue( input[i] );
	}
	
	// Check if the neuron has "fired" (ie: all inputs are full).
	if ( this.index != 0 ) return 0;
	
	// Calculate our value and propogate outwards.
	this.output = this.GetValue();
	
	// Iterate over the children.
	if ( this.Children !== undefined && this.Children.length > 0 ) {
		
		// If we are not an output node...
		for ( var i = 0; i < this.Children.length; i++ ) {
		
			// Evaluate
			var inputArray = new Array();
			inputArray.push( this.output );
			result += this.Children[i].Evaluate( inputArray );
		}
	} else {
		result = this.output;
	}
	
	return result;
}

Neuron.prototype.SetValue = function( input ) {
	this.error = 0;
	// Set the value in our array.
	this.inputs[this.index++] = input;
	// If we are in the "bias" territory, reset the index.
	if ( this.index > this.weights.length - 2 ) this.index = 0;
}

Neuron.prototype.GetValue = function() {

	var result = 0;
	
	// Iterate over each input.
	for ( var i = 0; i < this.inputs.length; i++ ) {
		// Calculate the value.
		result += this.inputs[i] * this.weights[i];
	}
	
	// Run the value through the activation function.
	this.output = MathHelper.Sigmoid( result );
	return this.output;
}

Neuron.prototype.GetChildWeight = function( index ) {	
	// This method will look at the [index] child and
	// find the weight that was used in a calculation based
	// on this current nodes output.
	if ( this.Children !== undefined && this.Children.length > index ) {
		for ( var r = 0; r < this.Children[index].weights.length; r++ ) {
			if ( this.Children[index].inputs[r] == this.output ) {
				return this.Children[index].weights[r];
			}
		}
	}
	
	// If no math is found, return one.
	return 1;
}

Neuron.prototype.CalculateError = function( target ) {
	// If we have already calculated the error, just return that.
	if ( this.error != 0 ) return this.error;
	
	// Do the actual calculation, otherwise.
	if ( this.Children === undefined || this.Children.length == 0 )  {
		// This is an output node, so we do a slightly different calculation.
		this.error = this.output * ( 1 - this.output ) * ( target - this.output );
	} else {
		// NOTE: We're basically taking the derivative of our
		// activation function and plugging in all the numbers. That's
		// why this math may look strange.
		this.error = this.output * ( 1 - this.output );
		var temp = 0;
		
		// Iterate over each child and calculate the error
		// associated with the weight that connects to it.
		for ( var i = 0; i < this.Children.length; i++ ) {
			// At it's simplest form, the error is equal to
			// the error of all the things below us times the weights that 
			// used our nodes output.
			temp += this.Children[i].CalculateError( target ) * this.GetChildWeight( i );
		}
		
		// So we calculate all those errors and multiply it to our own. Basically
		// the closer to the "top" of the graph you get, the less blamed
		// those nodes will be.
		this.error *= temp;
	}
	
	return this.error;
}

Neuron.prototype.Train = function() {
	// Iterate over our weights and update them
	// based on the error calculations.
	for ( var i = 0; i < this.weights.length; i++ ) { 
		this.weights[i] += this.LEARN_RATE * this.inputs[i] * this.error;
	}
}