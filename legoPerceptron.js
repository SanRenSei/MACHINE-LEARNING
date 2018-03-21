var fs = require('fs');
var {NeuralNetwork,LeakyRELUNeuralNetwork,CombinationNetwork,LeakyRELUCombinationNetwork} = require('./neuralNetworks.js');
var {LearningNetworkManager} = require('./neuralNetworkManager.js');

/*
Unit test for Neural Layer
var testLayer = new NeuralLayer(5,1);
console.log(testLayer.evaluate([1,2,3,4,5]));
console.log(testLayer.backpropagate([1]));
testLayer = new NeuralLayer(3,5);
console.log(testLayer.evaluate([1,2,3]));
console.log(testLayer.backpropagate([1,2,3,4,5]));
*/
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
//        Unit test for training XOR
/*
var testNet = new LeakyRELUNeuralNetwork([2,2,1]);
var testInputs = [
  [0,0],
  [0,1],
  [1,0],
  [1,1]
];
var testOutputs = [
  [0],
  [1],
  [1],
  [0]
];
var i;
for (i=0;i<500000;i++) {
  var testCase = Math.floor(4*Math.random());
  testNet.train(testInputs[testCase],testOutputs[testCase]);
}
var errorCalc = new LearningNetworkManager(testNet);
console.log(errorCalc.calculateError(testInputs,testOutputs));
testNet.printWeights();
*/
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
//    Unit testing for Rock Paper Scissors
/*
var things = [`Rock`,`Paper`,`Scissors`,`Knife`];
var testInputs = [
  [`Rock`,`Paper`],
  [`Rock`,`Scissors`],
  [`Rock`,`Rock`],
  [`Paper`,`Rock`],
  [`Paper`,`Scissors`],
  [`Paper`,`Paper`],
  [`Scissors`,`Rock`],
  [`Scissors`,`Paper`],
  [`Scissors`,`Scissors`]
];
var testOutputs = [
  [0,1],
  [1,0],
  [0.5,0.5],
  [1,0],
  [0,1],
  [0.5,0.5],
  [0,1],
  [1,0],
  [0.5,0.5]
];
var testNet = new LeakyRELUCombinationNetwork(things,2,[4,4,2]);
var i;
for (i=0;i<1000000;i++) {
  var testCase = Math.floor(testInputs.length*Math.random());
  testNet.train(testInputs[testCase],testOutputs[testCase]);
}
for (i=0;i<testInputs.length;i++) {
  console.log(testNet.evaluate(testInputs[i]));
}
testNet.printThingVectors();
*/
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
//Test For Curriculum Learning With Rock Paper Scissors Knife

var things = [];
var testInputs = [
  [`Rock`,`Paper`],
  [`Rock`,`Scissors`],
  [`Rock`,`Rock`],
  [`Paper`,`Rock`],
  [`Paper`,`Scissors`],
  [`Paper`,`Paper`],
  [`Scissors`,`Rock`],
  [`Scissors`,`Paper`],
  [`Scissors`,`Scissors`]
];
var testOutputs = [
  /*[0,1],
  [1,0],
  [0.5,0.5],
  [1,0],
  [0,1],
  [0.5,0.5],
  [0,1],
  [1,0],
  [0.5,0.5]*/
  [0,1,0],
  [1,0,0],
  [0,0,1],
  [1,0,0],
  [0,1,0],
  [0,0,1],
  [0,1,0],
  [1,0,0],
  [0,0,1]
];
var moreInputs = [
  [`Rock`,`Paper`],
  [`Rock`,`Scissors`],
  [`Rock`,`Knife`],
  [`Rock`,`Rock`],
  [`Paper`,`Rock`],
  [`Paper`,`Scissors`],
  [`Paper`,`Knife`],
  [`Paper`,`Paper`],
  [`Scissors`,`Rock`],
  [`Scissors`,`Paper`],
  [`Scissors`,`Scissors`],
  [`Scissors`,`Knife`],
  [`Knife`,`Rock`],
  [`Knife`,`Paper`],
  [`Knife`,`Knife`]
];
var moreOutputs = [
  [0,1,0],
  [1,0,0],
  [1,0,0],
  [0,0,1],
  [1,0,0],
  [0,1,0],
  [0,1,0],
  [0,0,1],
  [0,1,0],
  [1,0,0],
  [0,0,1],
  [0,0,1],
  [0,1,0],
  [1,0,0],
  [0,0,1]
];

var goodNetworkFound = false;

while (!goodNetworkFound) {
try {
var testNet = new LeakyRELUCombinationNetwork(things,2,[4,4,3]);
var testNetManager = new LearningNetworkManager(testNet);

testNetManager.supervisedTraining(testInputs,testOutputs,100000);
testNetManager.supervisedTraining(testInputs,testOutputs,100000);
testNetManager.supervisedTraining(testInputs,testOutputs,100000);
testNetManager.supervisedTraining(testInputs,testOutputs,100000);
testNetManager.supervisedTraining(testInputs,testOutputs,100000);
var error = testNetManager.calculateError(testInputs,testOutputs);
if (error>1) {
  throw 'Error too high.'
}
console.log('|||||||||||||||||||||||||||||||||||');
testNet.setNetworkLearning(0.0005);
testNet.setFittingLearning('Rock',0.001);
testNet.setFittingLearning('Paper',0.001);
testNet.setFittingLearning('Scissors',0.001);
testNetManager.supervisedTraining(moreInputs,moreOutputs,100000);
testNetManager.supervisedTraining(moreInputs,moreOutputs,100000);
testNetManager.supervisedTraining(moreInputs,moreOutputs,100000);
testNetManager.supervisedTraining(moreInputs,moreOutputs,100000);
testNetManager.supervisedTraining(moreInputs,moreOutputs,100000);
console.log(testNet.evaluate([`Knife`,`Scissors`]));

var testNetObj = testNet.toObj();
console.log(JSON.stringify(testNetObj));
testNet.fromObj(testNetObj);
testNetObj = testNet.toObj();
console.log(JSON.stringify(testNetObj));

error = testNetManager.calculateError(moreInputs,moreOutputs);
console.log(error);
if (error<=1 && !goodNetworkFound) {
  console.log('Try to write the model to file.');
  fs.writeFileSync('goodNetwork.json', JSON.stringify(testNetObj));
  console.log('Neural network logged');
  goodNetworkFound = true;
}
} catch (e) {}
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
