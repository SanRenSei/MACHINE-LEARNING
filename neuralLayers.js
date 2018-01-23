var {containsNaN} = require('./util.js');

function NeuralLayer(numInput, numOutput, learningRate = 0.01) {
  return new SigmoidNeuralLayer(numInput, numOutput, learningRate);
}

function SigmoidNeuralLayer(numInput, numOutput, learningRate = 0.01) {

  numInput++;
  this.inputNeurons = new Array(numInput);
  this.outputNeurons = new Array(numOutput);
  this.weights = new Array((numInput)*numOutput);
  var i;
  for (i=0;i<this.weights.length;i++) {
    this.weights[i] = 2*Math.random()-1;
  }
  this.learningRate = learningRate;
  
  this.evaluate = (input) => {
    var i;
    if (input.length!=(this.inputNeurons.length-1)) {
      throw `Expected ${this.inputNeurons.length-1} input values but received ${input.length}`;
    }
    this.inputNeurons = input.slice();
    this.inputNeurons.push(1);
    for (i=0;i<this.outputNeurons.length;i++) {
      this.outputNeurons[i]=0;
    }
    for (i=0;i<this.weights.length;i++) {
      this.outputNeurons[i%this.outputNeurons.length]+=this.weights[i]*this.inputNeurons[Math.floor(i/this.outputNeurons.length)];
    }
    for (i=0;i<this.outputNeurons.length;i++) {
      this.outputNeurons[i]=1/(1+Math.exp(-this.outputNeurons[i]));
    }
    return this.outputNeurons;
  };
  
  this.backpropagate = (deltaOut) => {
    var i;
    if (deltaOut.length!=this.outputNeurons.length) {
      throw `Expected ${this.outputNeurons.length} delta values but received ${deltaOut.length}`;
    }
    var deltaIn = [];
    for (i=0;i<this.inputNeurons.length;i++) {
      deltaIn[i]=0;
    }
    for (i=0;i<this.weights.length;i++) {
      deltaIn[Math.floor(i/this.outputNeurons.length)]+=this.weights[i]*deltaOut[i%this.outputNeurons.length];
    }
    for (i=0;i<this.inputNeurons.length;i++) {
      deltaIn[i]*=(this.inputNeurons[i]*(1-this.inputNeurons[i]));
    }
    for (i=0;i<this.weights.length;i++) {
      this.weights[i]-=this.learningRate*deltaOut[i%this.outputNeurons.length]*this.inputNeurons[Math.floor(i/this.outputNeurons.length)];
    }
    deltaIn.splice(deltaIn.length-1,1);
    return deltaIn;
  };
  return this;
}

function LeakyRELUNeuralLayer(numInput = 1, numOutput = 1, learningRate = 0.0001) {

  numInput++;
  this.inputNeurons = new Array(numInput);
  this.outputNeurons = new Array(numOutput);
  this.weights = new Array((numInput)*numOutput);
  var i;
  for (i=0;i<this.weights.length;i++) {
    this.weights[i] = 2*Math.random()-1;
  }
  this.learningRate = learningRate;
  
  this.evaluate = (input) => {
    var i;
    if (containsNaN(input)) {
      throw input;
    }
    if (input.length!=(this.inputNeurons.length-1)) {
      throw `Expected ${this.inputNeurons.length-1} input values but received ${input.length}`;
    }
    this.inputNeurons = input.slice();
    this.inputNeurons.push(1);
    for (i=0;i<this.outputNeurons.length;i++) {
      this.outputNeurons[i]=0;
    }
    for (i=0;i<this.weights.length;i++) {
      this.outputNeurons[i%this.outputNeurons.length]+=this.weights[i]*this.inputNeurons[Math.floor(i/this.outputNeurons.length)];
    }
    for (i=0;i<this.outputNeurons.length;i++) {
      this.outputNeurons[i]=((this.outputNeurons[i]>0)?(this.outputNeurons[i]):(this.outputNeurons[i]/100));
    }
    if (containsNaN(this.outputNeurons)) {
      throw this.outputNeurons;
    }
    return this.outputNeurons;
  };
  
  this.backpropagate = (deltaOut) => {
    if (containsNaN(deltaOut)) {
      throw deltaOut;
    }
    var i;
    if (deltaOut.length!=this.outputNeurons.length) {
      throw `Expected ${this.outputNeurons.length} delta values but received ${deltaOut.length}`;
    }
    var deltaIn = [];
    for (i=0;i<this.inputNeurons.length;i++) {
      deltaIn[i]=0;
    }
    for (i=0;i<this.weights.length;i++) {
      deltaIn[Math.floor(i/this.outputNeurons.length)]+=this.weights[i]*deltaOut[i%this.outputNeurons.length];
    }
    if (containsNaN(deltaIn)) {
      throw JSON.stringify({deltaIn,weights:this.weights,deltaOut});
    }
    for (i=0;i<this.inputNeurons.length;i++) {
      deltaIn[i]*=((this.inputNeurons[i]>0)?1:0.01);
    }
    for (i=0;i<this.weights.length;i++) {
      this.weights[i]-=this.learningRate*deltaOut[i%this.outputNeurons.length]*this.inputNeurons[Math.floor(i/this.outputNeurons.length)];
    }
    deltaIn.splice(deltaIn.length-1,1);
    if (containsNaN(deltaIn)) {
      throw deltaIn;
    }
    return deltaIn;
  };
  
  this.toObj = () => {
    var {inputNeurons, outputNeurons, weights} = this;
    return {inputNeurons, outputNeurons, weights};
  }
  
  this.fromObj = (obj) => {
    var {inputNeurons, outputNeurons, weights} = obj;
    this.inputNeurons = inputNeurons;
    this.outputNeurons = outputNeurons;
    this.weights = weights;
    return this;
  }
  
  return this;
}

exports.NeuralLayer = NeuralLayer;
exports.SigmoidNeuralLayer = SigmoidNeuralLayer;
exports.LeakyRELUNeuralLayer = LeakyRELUNeuralLayer;