var {NeuralLayer,SigmoidNeuralLayer,LeakyRELUNeuralLayer} = require('./neuralLayers.js');

function NeuralNetwork(layerCounts) {
  
  this.neuralLayers = [];
  var i;
  for (i=0;i<layerCounts.length-1;i++) {
    this.neuralLayers.push(new NeuralLayer(layerCounts[i],layerCounts[i+1]));
  }
  
  this.evaluate = (input) => {
    var i;
    var output = input;
    for (i=0;i<this.neuralLayers.length;i++) {
      output = this.neuralLayers[i].evaluate(output);
    }
    return output;
  };
  
  this.train = (input,target) => {
    var result = this.evaluate(input);
    var outputDelta = [];
    var i;
    for (i=0;i<result.length;i++) {
      outputDelta.push((result[i]-target[i])*(result[i])*(1-result[i]));
    }
    for (i=this.neuralLayers.length-1;i>=0;i--) {
      outputDelta = this.neuralLayers[i].backpropagate(outputDelta);
    }
  };
  
  this.printWeights = () => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      console.log(this.neuralLayers[i].weights);
    }
  }
  
}

function LeakyRELUNeuralNetwork(layerCounts) {
  
  this.neuralLayers = [];
  var i;
  for (i=0;i<layerCounts.length-1;i++) {
    this.neuralLayers.push(new LeakyRELUNeuralLayer(layerCounts[i],layerCounts[i+1]));
  }
  
  this.evaluate = (input) => {
    var i;
    var output = input;
    for (i=0;i<this.neuralLayers.length;i++) {
      output = this.neuralLayers[i].evaluate(output);
    }
    return output;
  };
  
  this.train = (input,target) => {
    var result = this.evaluate(input);
    var outputDelta = [];
    var i;
    for (i=0;i<result.length;i++) {
      outputDelta.push((result[i]-target[i])*(result[i]>0?1:0.01));
    }
    for (i=this.neuralLayers.length-1;i>=0;i--) {
      outputDelta = this.neuralLayers[i].backpropagate(outputDelta);
    }
  };
  
  this.printWeights = () => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      console.log(this.neuralLayers[i].weights);
    }
  }
  
}

function CombinationNetwork(things,detailsPerThing,layerCounts) {
  
  this.things = things;
  this.thingLayers = {};
  things.forEach(thing => {
    this.thingLayers[thing] = new NeuralLayer(0,detailsPerThing,0.1);
  });
  this.neuralLayers = [];
  var i;
  for (i=0;i<layerCounts.length-1;i++) {
    this.neuralLayers.push(new NeuralLayer(layerCounts[i],layerCounts[i+1],0.025));
  }
  
  this.processInputs = (inputs) => {
    var processedInputs = [];
    var i;
    for (i=0;i<inputs.length;i++) {
      var input = inputs[i];
      if (typeof input == `string`) {
        var thingVector = this.thingLayers[input].evaluate([]);
        processedInputs = processedInputs.concat(thingVector);
      } else {
        processedInputs.push(input);
      }
    }
    return processedInputs;
  };
  
  this.evaluate = (input) => {
    var i;
    input = this.processInputs(input);
    var output = input;
    for (i=0;i<this.neuralLayers.length;i++) {
      output = this.neuralLayers[i].evaluate(output);
    }
    return output;
  };
  
  this.train = (input,target) => {
    var result = this.evaluate(input);
    var outputDelta = [];
    var i;
    for (i=0;i<result.length;i++) {
      outputDelta.push((result[i]-target[i])*(result[i])*(1-result[i]));
    }
    for (i=this.neuralLayers.length-1;i>=0;i--) {
      outputDelta = this.neuralLayers[i].backpropagate(outputDelta);
    }
    for (i=0;i<outputDelta;i++) {
      if (typeof input[i] == `string`) {
        var thingDelta = outputDelta.splice(i,detailsPerThing,0);
        this.thingLayers[input[i]].backpropagate(thingDelta);
      }
    }
  };
  
  this.holdNetwork = () => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      this.neuralLayers[i].learningRate = 0;
    }
  };
  
  this.reduceFitting = (thing) => {
    this.thingLayers[thing].learningRate = 0.025;
  };
  
  this.printWeights = () => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      console.log(this.neuralLayers[i].weights);
    }
  };
  
  this.printThingVectors = () => {
    var things = Object.keys(this.thingLayers);
    var i;
    for (i=0;i<things.length;i++) {
      console.log(things[i]);
      console.log(this.thingLayers[things[i]].evaluate([]));
    }
  };
  
}

function LeakyRELUCombinationNetwork(things = [],detailsPerThing = 1,layerCounts = []) {
  
  this.things = things;
  this.thingLayers = {};
  things.forEach(thing => {
    this.thingLayers[thing] = new LeakyRELUNeuralLayer(0,detailsPerThing,0.01);
  });
  this.neuralLayers = [];
  var i;
  for (i=0;i<layerCounts.length-1;i++) {
    this.neuralLayers.push(new LeakyRELUNeuralLayer(layerCounts[i],layerCounts[i+1],0.01));
  }
  
  this.processInputs = (inputs) => {
    var processedInputs = [];
    var i;
    for (i=0;i<inputs.length;i++) {
      var input = inputs[i];
      if (typeof input == `string`) {
        var thingVector = this.thingLayers[input].evaluate([]);
        processedInputs = processedInputs.concat(thingVector);
      } else {
        processedInputs.push(input);
      }
    }
    return processedInputs;
  };
  
  this.evaluate = (input) => {
    var i;
    input = this.processInputs(input);
    var output = input;
    for (i=0;i<this.neuralLayers.length;i++) {
      output = this.neuralLayers[i].evaluate(output);
    }
    return output;
  };
  
  this.train = (input,target) => {
    var result = this.evaluate(input);
    var outputDelta = [];
    var i;
    for (i=0;i<result.length;i++) {
      outputDelta.push((result[i]-target[i])*(result[i]>0?result[i]:(result[i]/100)));
    }
    for (i=this.neuralLayers.length-1;i>=0;i--) {
      outputDelta = this.neuralLayers[i].backpropagate(outputDelta);
    }
    for (i=0;i<outputDelta;i++) {
      if (typeof input[i] == `string`) {
        var thingDelta = outputDelta.splice(i,detailsPerThing,0);
        this.thingLayers[input[i]].backpropagate(thingDelta);
      }
    }
  };
  
  this.setNetworkLearning = (rate) => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      this.neuralLayers[i].learningRate = rate;
    }
  };
  
  this.setFittingLearning = (thing, rate) => {
    this.thingLayers[thing].learningRate = rate;
  };
  
  this.printWeights = () => {
    var i;
    for (i=0;i<this.neuralLayers.length;i++) {
      console.log(this.neuralLayers[i].weights);
    }
  };
  
  this.printThingVectors = () => {
    var things = Object.keys(this.thingLayers);
    var i;
    for (i=0;i<things.length;i++) {
      console.log(things[i]);
      console.log(this.thingLayers[things[i]].evaluate([]));
    }
  };
  
  this.toObj = () => {
    var {things,thingLayers,neuralLayers} = this;
    thingLayers = things.map(t=>thingLayers[t]);
    neuralLayers = neuralLayers.map(l=>l.toObj());
    return {things, thingLayers, neuralLayers};
  };
  
  this.fromObj = (obj) => {
    var {things,thingLayers,neuralLayers} = obj;
    this.things = things;
    this.thingLayers = {};
    var i;
    for (i=0;i<things.length;i++) {
      this.thingLayers[things[i]]=thingLayers[i];
    }
    this.neuralLayers = neuralLayers.map(o=>(new LeakyRELUNeuralLayer()).fromObj(o));
    return this;
  };
  
}

exports.NeuralNetwork = NeuralNetwork;
exports.LeakyRELUNeuralNetwork = LeakyRELUNeuralNetwork;
exports.CombinationNetwork = CombinationNetwork;
exports.LeakyRELUCombinationNetwork = LeakyRELUCombinationNetwork;
