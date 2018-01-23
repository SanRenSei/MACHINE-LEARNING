"use strict";

var i,j,k,l,m;
var nLayer=[];
var values=[];
var weights=[];

var learningRate = 0.5;

function initPerceptron(layers) {

	for (i=0;i<layers.length;i++) {
		nLayer[i]=layers[i];
	}

	for (i=0;i<layers.length;i++) {
		values[i]=[];
		for (j=0;j<nLayer[i]+1;j++) {
			values[i][j]=0;
		}
	}

	for (i=0;i<layers.length-1;i++) {
		weights[i]=[];
		for (j=0;j<nLayer[i+1]+1;j++) {
			weights[i][j]=[];
			for (k=0;k<nLayer[i]+1;k++) {
				weights[i][j][k]=0;
			}
		}
	}

	generateRandomWeights();
}

function setLearningRate(lr) {
	learningRate = lr;
}


function generateRandomWeights() {

	for (k=0;k<nLayer.length-1;k++) {
		for(j=1;j<=nLayer[k+1];j++) {
			for(i=0;i<=nLayer[k];i++) {
				weights[k][j][i] = Math.random() - 0.5;
			}
		}
	}
}

function train(pattern, desiredOutput) {
	var output = passNet(pattern);
	backpropagation(desiredOutput);

	return output;
}

function passNet(pattern) {

	for(i=0;i<nLayer[0];i++) {
		values[0][i+1] = pattern[i];
	}

	// Set bias
	for (i=0;i<nLayer.length-1;i++) {
		values[i][0] = 1;
	}

	for (k=0;k<nLayer.length-1;k++) {
		for(j=1; j<=nLayer[k+1];j++) {
			values[k+1][j] = 0;
			for(i=0;i<=nLayer[k];i++) {
				values[k+1][j] += weights[k][j][i] * values[k][i];
			}
			values[k+1][j] = 1/(1+Math.exp(-values[k+1][j]));
		}
	}

	return values[nLayer.length-1];
}

function backpropagation(desiredOutput) {

	var errors=[];
	for (i=0;i<nLayer.length-1;i++) {
		errors[i]=[];
		for (j=0;j<nLayer[i+1]+1;j++) {
			errors[i][j]=0;
		}
	}

	var Esum = 0;

	for(i=1;i<=nLayer[nLayer.length-1];i++)
		errors[nLayer.length-2][i] = values[nLayer.length-1][i] * (1-values[nLayer.length-1][i]) * (desiredOutput[i-1]-values[nLayer.length-1][i]);


	for (k=0;k<nLayer.length-2;k++) {
		for(i=0; i<=nLayer[k+1]; i++) {
			for(j=1; j<=nLayer[k+2]; j++)
				Esum += weights[k+1][j][i] * errors[k+1][j];

			errors[k][i] = values[k+1][i] * (1-values[k+1][i]) * Esum;
			Esum = 0;
		}
	}

	for (k=nLayer.length-1;k>0;k--) {
		for(j=1; j<=nLayer[k]; j++)
			for(i=0; i<=nLayer[k-1]; i++)
				weights[k-1][j][i] += learningRate * errors[k-1][j] * values[k-1][i];
	}
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////


//Using Perceptron

var layerCount = [2,5,5,1];
initPerceptron(layerCount);
var inputs = [
		[0,0],
		[0,1],
		[1,0],
		[1,1]
];
var outputs = [
		[0],
		[1],
		[1],
		[0]
];
var cycles = 10000000;
for (l=0;l<cycles;l++) {
	if (l%(cycles/100)==0) {
		console.log(l*100/cycles+"%");
	}
	m = Math.floor(Math.random()*4);
		train(inputs[m], outputs[m]);
}
console.log(passNet(inputs[0])[1]);
console.log(passNet(inputs[1])[1]);
console.log(passNet(inputs[2])[1]);
console.log(passNet(inputs[3])[1]);
