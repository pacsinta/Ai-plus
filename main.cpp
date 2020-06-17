#define input_count 10
#define trainingData_count 0
#define layer_count 3
#define mnist_location "mnist/"


#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>

#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"



//regularizer is missing


enum class LossFunction {
	//regression loss functions
	MeanSquare = 0,  //working progress
	MeanSquareLogarithmic = 1,
	MeanAbsoluteError = 2,

	//binary classification loss functions
	HingeLoss = 3, //working progress
	BinaryCrossEntropy = 4,
	SquaredHingeLoss = 5,

	//multy-class classification loss functions
	MultyClassCrossEntropy = 6,
	SparseMultiClassCrossEntropy = 7,
	KullbackLeiblerDivergenceLoss = 8
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

        //Generate random float
		float randomFloat(float min, float max) {
			return (min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min))));
		}

        //Return x^2
		float squareFloat(float x) {
			return x * x;
		}
};




class Log {
public:
    //Write all data from a float array to console
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
	Log log;
public:
	float* weight;
	float output;  //Output with activation function
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
	void Train(int target, float* prevOutput, int n_input, float learningRate, bool outputLayer, float* oldG, int G_count, float* oldWeights, LossFunction lf){
		//only for tanh

		switch(lf)
		{
        case LossFunction::MeanSquare:
            switch(activation)
            {
            case Functions::Tanh:
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
            case Functions::Sigmoind:
                if (outputLayer)
                {

                    G = (output-target)*(output-math.squareFloat(output));
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
                    G = G * (output-math.squareFloat(output));



                    for (int i = 0; i < n_input; i++)
                    {
                        weight[i] = weight[i] - learningRate * G * prevOutput[i];
                    }

                }
            }
        case LossFunction::HingeLoss:
            switch(activation)
            {
                case Functions::Sigmoind:
                    break;
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
	Functions f;
public:
	NeuralNetwork(int* layers, float* n_biases, LossFunction n_lf, Functions n_f) {

        lf = n_lf;
        f=n_f;
		p_layers = layers;
		network.resize(layer_count);
		biases = n_biases;

		network[0].resize(p_layers[0]);
		for (int i = 0; i < p_layers[0]; i++)
		{
			network[0][i].init(1, f, biases[0]);
		}

		for (int i = 0; i < layer_count; i++)
		{
			network[i].resize(p_layers[i]);

			for (int x = 0; x < p_layers[i]; x++)
			{
				network[i][x].init(p_layers[i - 1], f, biases[i]);
			}
		}

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
		}

		return outputs[layer_count-1];
	}

	void Train(float** trainInput, int inputCount, float** trainOutput, int trainDataCount, float learningRate, int learningCount, int* layers, unsigned int epochs) {
		for(int a=0; a<epochs; a++){
            for(int b=0; b<trainDataCount; b++){
                Run(trainInput[b]);

                //train Output layer
                for (int i = 0; i < layers[layer_count-1]; i++)
                {
                    float* prevOutput=new float[layers[layer_count-2]];
                    for(int x = 0; x < layers[layer_count-2]; x++)
                    {
                        prevOutput[x] = network[layer_count-2][x].output;
                    }
                    network[layer_count-1][i].Train(trainOutput[b][i], prevOutput, layers[layer_count-2], learningRate, true, nullptr, 0, nullptr, lf);
                }

                for (int i = layer_count-2; i > -1; i--)
                {
                    if(i==0){
                        float* prevOutput=new float[inputCount];
                        for(int x = 0; x < inputCount; x++)
                        {
                            prevOutput[x] = trainInput[b][x];
                        }
                    }else{
                        float* prevOutput=new float[layers[i-1]];
                        for(int x = 0; x < layers[i-1]; x++)
                        {
                            prevOutput[x] = network[i-1][x].output;
                        }

                        float* oldG=new float[layers[i+1]];
                        float* oldweight=new float[layers[i+1]];
                        for(int x = 0; x < layers[i+1]; x++){
                            oldG[x] = network[i+1][x].G;
                        }

                        for (int x = 0; x < layers[i]; x++)
                        {
                            for(int z=0; z<layers[i+1]; z++){
                                oldweight[z]=network[i+1][z].weight[x];
                            }
                            network[i][x].Train(0, prevOutput, layers[i-1], learningRate, false, oldG, layers[i+1], oldweight, lf);
                        }
                    }
                }
            }
		}
	}

	void Test(float** testInputs, float** testOutput, int testDataCount, int* layers){
        float averageloss=0;
        Run(testInputs[0]);

        switch(lf){
        case LossFunction::MeanSquare:
            for(int i=0; i<layers[layer_count-1]; i++){

            }
        }
	}

	~NeuralNetwork() {   //destructor

	}

};



int main() {
	std::cout << "Hello Ai" << std::endl;

    /*
	int* layers = new int[layer_count] {2, 3, 2};
	float* biases = new float[layer_count] {1, 1, 1};

	NeuralNetwork nn(layers, biases);

	float** train_in = new float* [1];
	train_in[0] = new float[layers[0]]{ 1.0f, 2.0f };

	float** train_out = new float* [1];
	train_out[0] = new float[layers[layer_count-1]]{ 1.0f , 0.1f};


	nn.Train(train_in, 2, train_out, 1, 1, 1, LossFunction::MeanSquare, layers, 1);
    */


	std::cout << "Load mnist data" << std::endl;

	mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(mnist_location);
    mnist::normalize_dataset(dataset);

    std::cout<<dataset.training_images.size() << std::endl;;
    std::cout<< dataset.training_images[0][0];


    int* layers = new int[layer_count]{784, 50, 10};
    float* biases = new float[layer_count] {1, 1, 1};

    NeuralNetwork nn(layers, biases, LossFunction::HingeLoss, Functions::Sigmoind);

    float** train_in = new float*[dataset.training_images.size()];
    float** train_out = new float*[dataset.training_images.size()];


    for(int i=0; i<dataset.training_images.size(); i++){
        train_in[i]=&dataset.training_images[i][0];
        train_out[i] = new float[1];
        train_out[i][0]=(float)dataset.training_labels[i];
        std::cout<<train_out[i][0]<<std::endl;
    }


    std::cout<<"Train"<<std::endl;
    nn.Train(train_in, 784, train_out, dataset.training_images.size(), 0.1, 100, layers, 5);


    delete[] train_in;
    delete[] train_out;
    std::cout<<"Finished";
}
