#define input_count 10 
#define trainingData_count 0
#define layer_count 3


#include <iostream>
#include <math.h>
#include <vector>

enum class LossFunction {
	//regression losses
	MeanSquare = 0,

	//classification losses
	HingeLoss = 1,
	Simple = 2
};

enum class Functions {
	Sigmoind = 0,  //used to predict propability
	Tanh = 1,
	//HyperBolicTangent,
	ReLu = 2, 
	Identity = 3,
	Step = 4
};

class Math {
	public:
		float randomFloat(float min, float max) {
			return (min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min))));
		}
		float squareFloat(float x) {
			return x * x;
		}
};




class Log {
public: 
	void logFloatPointerArray(float* arr, int size) {
		std::ios_base::sync_with_stdio(false);

		for (int i = 0; i < size; i++)
		{
			std::cout << arr[i] << std::endl;
		}
	}
};


class Neuron {
private:
	Math math;
	Functions activation;
	float bias;
public:
	float* weight;
	float output;
	float bOutput; //before activation output
	float G; //buffer for backprop
	void init(unsigned int n_input, Functions n_activation, float n_bias) {
		weight = new float[n_input];
		for (unsigned int i = 0; i < n_input; i++)
		{
			weight[i] = math.randomFloat(-1.0f, 1.0f);
			//std::cout << weight[i] << " ";
		}
		//std::cout << std::endl;
		activation = n_activation;
		bias = n_bias;
	}
	void Run(float* inputs, unsigned int n_input) {
		bOutput = 0;
		
		for (unsigned int i = 0; i < n_input; i++)
		{
			bOutput += inputs[i] * weight[i];
		}
		
		bOutput += bias;
		
		switch (activation)
		{
		case Functions::Sigmoind:
			output = (1.0f / (1.0f + (float)exp(-bOutput)));
		case Functions::ReLu:
			if (bOutput < 0)
			{
				output = 0;
			}
			else
			{
				output = bOutput;
			}
		case Functions::Identity:  //missing
			output = 0;
		case Functions::Step:
			if (bOutput > 0)
			{
				output = 1;
			}
			else
			{
				output = 0;
			}
		case Functions::Tanh:
			output = ((float)exp(bOutput)-(float)exp(-bOutput))/((float)exp(bOutput)+(float)exp(-bOutput));
		}
	}
	void Train(int target, float* prevOutput, int n_input, float learningRate, bool outputLayer, float* oldG, int G_count, float* oldWeights){
		//only for tanh
		
		if (outputLayer)
		{
			
			G = (output-target)*(1-math.squareFloat(output));
			for (int i = 0; i < n_input; i++)
			{
				weight[i] = weight[i] - learningRate * G * prevOutput[i];
			}
		}else
		{
			for (int i = 0; i < G_count; i++)
			{
				G += oldG[i] * oldWeights[i];
			}
			G = G * (1-math.squareFloat(output));
			for (int i = 0; i < n_input; i++)
			{
				weight[i] = weight[i] - learningRate * G * prevOutput[i];
			}
		}
	}
	
	~Neuron() {
		delete[] weight;
	}
};




class NeuralNetwork {
private:
	std::vector<std::vector<Neuron>> network;
	int* p_layers;
	LossFunction lf;
	Math math;
	Log log;
	float* biases;
public:
	NeuralNetwork(int* layers, float* n_biases) {
		
		p_layers = layers;
		network.resize(layer_count);
		biases = n_biases;

		network[0].resize(p_layers[0]);
		for (int i = 0; i < p_layers[0]; i++)
		{
			network[0][i].init(1, Functions::Tanh, biases[0]);
		}

		for (int i = 0; i < layer_count; i++)
		{
			network[i].resize(p_layers[i]);

			for (int x = 0; x < p_layers[i]; x++)
			{
				network[i][x].init(p_layers[i - 1], Functions::Tanh, biases[i]);
			}
		}
		lf = LossFunction::HingeLoss;
	}
	float* Run(float* inputs) {

		float* input_buff;
		float** outputs = new float*[layer_count];

		outputs[0] = new float[p_layers[0]];
		for (int i = 0; i < p_layers[0]; i++)
		{
			input_buff = &inputs[i];
			network[0][i].Run(input_buff, 1);
			outputs[0][i] = network[0][i].output;
		}



		for (int i = 1; i < layer_count; i++)
		{
			outputs[i] = new float[p_layers[i]];
			//std::cout << (p_layers[i] - 1) << std::endl;
			for (int x = 0; x < p_layers[i]; x++)
			{
				network[i][x].Run(outputs[i-1], p_layers[i - 1]);
				outputs[i][x] = network[i][x].output;
			}
			//Log log;
			//log.logFloatPointerArray(outputs[i], p_layers[i]);
		}

		return outputs[layer_count-1];
	}

	void Train(float** trainInput, int inputCount, float** trainOutput, int trainDataCount, float learningRate, int learningCount, LossFunction lossFunction, int* layers, unsigned int batches) {
		Run(trainInput[0]);

		//train Output layer
		for (int i = 0; i < layers[layer_count-1]; i++)
		{
			float* prevOutput=new float[layers[layer_count-2]];
			for(int x = 0; x < layers[layer_count-2]; x++)
			{
				prevOutput[x] = network[layer_count-2][x].output;
			}
			network[layer_count-1][i].Train(trainOutput[0][i], prevOutput, layers[layer_count-2], learningRate, true, nullptr, 0, nullptr);
		}	
		
		for (int i = layer_count-2; i > -1; i--)
		{
			if(i==0){
				float* prevOutput=new float[inputCount];
				for(int x = 0; x < inputCount; x++)
				{
					prevOutput[x] = trainInput[0][x];
				}
			}else{
				float* prevOutput=new float[layers[i-1]];
				for(int x = 0; x < layers[i-1]; x++)
				{
					prevOutput[x] = network[i-1][x].output;
				}
				float* prevG=new float[layers[i+1]];
				for(int x = 0; x < layers[i+1]; x++){
					prevG[x] = network[i+1][x].G;
				}

				for (int x = 0; x < layers[i]; x++)
				{
					network[i][x].Train(0, prevOutput, layers[i-1], learningRate, false, prevG, layers[i+1], network[i+1][x].weight);
				}
			}
		}
	}

	~NeuralNetwork() {   //destructor

	}
	
};


int main() {
	std::cout << "Hello Ai" << std::endl;
	int* layers = new int[layer_count] {2, 3, 1};
	float* biases = new float[layer_count] {1, 1, 1};

	NeuralNetwork nn(layers, biases);

	float** train_in = new float* [1];
	train_in[0] = new float[2]{ 1.0f, 2.0f };

	float** train_out = new float* [1];
	train_out[0] = new float[1]{ 1.0f };


	nn.Train(train_in, 2, train_out, 1, 1, 1, LossFunction::MeanSquare, layers, 1);
}
