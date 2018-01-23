

function LearningNetworkManager(network) {
  
  this.network = network;
  
  this.calculateError = (inputs,outputs) => {
    var totalError = 0;
    var i;
    for (i=0;i<inputs.length;i++) {
      var input = inputs[i];
      var output = outputs[i];
      var networkOut = network.evaluate(input);
      var j;
      for (j=0;j<output.length;j++) {
        var difference = output[j]-networkOut[j];
        totalError += difference*difference;
      }
    }
    return totalError;
  };
  
  this.supervisedTraining = (inputs, outputs, steps = 100000) => {
    var error = 999999;
    var newError = this.calculateError(inputs,outputs);
    while (newError<error && newError>0.000001) {
      error = newError;
      console.log(error);
      for (i=0;i<steps;i++) {
        var testCase = Math.floor(inputs.length*Math.random());
        this.network.train(inputs[testCase],outputs[testCase]);
      }
      newError = this.calculateError(inputs,outputs);
    }
    console.log(newError);
  };
  
  this.getNetworkObj = () => {
    return network.toObj();
  };
  
  return this;
  
}

exports.LearningNetworkManager = LearningNetworkManager;