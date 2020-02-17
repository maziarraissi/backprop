/*********************
@author: Maziar Raissi
*********************/

/****************************************************
To compile and run use:
    nvcc -std=c++11 CudaSimpleNN.cu -o CudaSimpleNN
    ./CudaSimpleNN
****************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <chrono>

#include <cmath>
#include <cstdio>

using namespace std;
using namespace std::chrono;

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 1  ////////////////////////////////////
//////////////////// Matrix Algebra /////////////////////////////////
/////////////////////////////////////////////////////////////////////

// macro
#define idx(i,j,m) (((j)*(m))+(i))

// print matrix
void print(float* mat, int m, int n)
{
	cout << "\n";
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
			cout << mat[idx(i,j,m)] << " ";
		cout << "\n";
	}
	cout << "\n";
}

// squared l2 error between two matrices of size n x q
float relative_l2_error(float* Y_pred, float* Y, int n, int q)
{
	float loss = 0;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < q; ++j)
			loss += 0.5*pow(Y_pred[idx(i,j,n)] - Y[idx(i,j,n)],2);
	return loss;
}

// relative l2 error between two matrices of size n x q
float squared_l2_error(float* Y_pred, float* Y, int n, int q)
{
	float error_numerator = 0;
	float error_denominator = 0;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < q; ++j)
		{
			error_numerator += pow(Y_pred[idx(i,j,n)] - Y[idx(i,j,n)],2);
			error_denominator += pow(Y[idx(i,j,n)],2);
		}

	return sqrt(error_numerator/error_denominator);
}

// copy a matrix of sizes n x p
__global__ void copy(float* input, float* output1, float* output2)
{

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;
	// int p = blockDim.x;

	output1[idx(i,j,n)] = input[idx(i,j,n)];
	output2[idx(i,j,n)] = input[idx(i,j,n)];
}

// Dense Layer
// A = tanh(H), (n x q) = (n x q)
// H=X*W+b, (n x q) = (n x p)*(p x q)+(1 x q)
__global__ void dense(float* X, float* W, float* b, float* H, float* A, int p)
{

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;
	// int q = blockDim.x;
	
	H[idx(i,j,n)] = b[idx(0,j,1)];
	for (int k = 0; k < p; ++k)
		H[idx(i,j,n)] += X[idx(i,k,n)] * W[idx(k,j,p)];
	
	A[idx(i,j,n)] = tanh(H[idx(i,j,n)]);
}

// (res = lhs^T*rhs) matrix multiplication of two matrices of sizes n x p and n x q 
__global__ void matmul1(float* lhs, float* rhs, float* res, int n)
{

	int i = blockIdx.x;
	int j = threadIdx.x;
	int p = gridDim.x;
	// int q = blockDim.x;
	
	res[idx(i,j,p)] = 0;
	for (int k = 0; k < n; ++k)
		res[idx(i,j,p)] += lhs[idx(k,i,n)] * rhs[idx(k,j,n)];
}

// backpropagation step: G0 = (1-A.A).(G1*W^T), (n x p) = (1 - (n x p).(n x p)).((n x q)*(p x q)^T)
__global__ void backprop(float* A, float* G1, float* W, float* G0, int q)
{

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;
	int p = blockDim.x;
	
	G0[idx(i,j,n)] = 0;
	for (int k = 0; k < q; ++k)
		G0[idx(i,j,n)] += G1[idx(i,k,n)] * W[idx(j,k,p)];
	G0[idx(i,j,n)] = (1.0-A[idx(i,j,n)]*A[idx(i,j,n)])*G0[idx(i,j,n)];
}

// sum the rows of a matrix of size m x n
__global__ void row_sum(float* input, float* output, int m)
{
	// int i = blockIdx.x; // i = 0
	int j = threadIdx.x; // j = 0,1,...,n-1
	// int m = gridDim.x; // m = 1
	// int n = blockDim.x; // n

	output[idx(0,j,1)] = 0;
	for (int i = 0; i < m; ++i)
		output[idx(0,j,1)] += input[idx(i,j,m)];
}

// pointwise subtraction on two matrices of size n x q
__global__ void sub(float* lhs, float* rhs, float* res)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = gridDim.x;
	// int q = blockDim.x;
	
	res[idx(i,j,n)] = lhs[idx(i,j,n)] - rhs[idx(i,j,n)];
}


/////////////////////////////////////////////////////////////////////
//////////////////////// Part 2  ////////////////////////////////////
/////////////////// Helper Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////

// Function to Load Data
tuple< vector<float>, vector<float>, int > load_data(string file_name)
{
	ifstream ifs {file_name};
	if (!ifs)
		cerr << "Could not open file:" << file_name << "\n";

	vector<float> X;
	vector<float> Y;
	string line;
	while (getline(ifs, line))
	{
		float x, y;
		istringstream line_stream(line);
		line_stream >> x >> y;
		X.push_back(x);
		Y.push_back(y);
	}
	
	int N = X.size();

	return make_tuple(X, Y, N);
}

// Xavier Initializer
void xavier_init(int in_dim, int out_dim, float* W, float* b)
{
    float xavier_stddev = sqrt(2.0/(in_dim + out_dim));

	int seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
    normal_distribution<float> distribution(0.0,xavier_stddev);
    
	for (int i = 0; i < in_dim; ++i)
		for (int j = 0; j < out_dim; ++j)
		{
			W[idx(i,j,in_dim)] = distribution(generator);
			b[idx(0,j,1)] = 0.0f;
		}
}

// Adam Optimizer
__global__ void stochastic_update_Adam(float* w,
									   float* grad_w,
									   float* mt,
									   float* vt,
									   float lrate,
									   int iteration)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int m = gridDim.x;
	// int n = blockDim.x;

    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;

	mt[idx(i,j,m)] = beta1*mt[idx(i,j,m)] + (1.0-beta1)*grad_w[idx(i,j,m)];
    vt[idx(i,j,m)] = beta2*vt[idx(i,j,m)] + (1.0-beta2)*(grad_w[idx(i,j,m)]*grad_w[idx(i,j,m)]);

    float mt_hat = (1.0/(1.0-pow(beta1,iteration)))*mt[idx(i,j,m)];
    float vt_hat = (1.0/(1.0-pow(beta2,iteration)))*vt[idx(i,j,m)];

    float scal = 1.0/(epsilon + sqrt(vt_hat));

    w[idx(i,j,m)] = w[idx(i,j,m)] - (lrate*(mt_hat*scal));
	
}

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 3  ////////////////////////////////////
//////////////////// Neural Networks ////////////////////////////////
/////////////////////////////////////////////////////////////////////

// Neural Networks Class [ y = f(x) ]
class SimpleNN
{

private:

	float* X; // n x p
	float* Y; // n x q
	int n; // number of data
	int p; // dimension of x
	int q; // dimension of y

	vector<int> layers; // layers (e.g., {1,10,10,10,1})
	int num_layers;
	vector<float*> weights; // weights
	vector<float*> biases; // biases

	vector<float*> H_list; // hidden units
    vector<float*> A_list; // activated units

	vector<float*> G_list; // backpropagation units
	vector<float*> loss_weights; // gradients of the loss with respect to weights
	vector<float*> loss_biases; // gradients of the loss with respect to biases

	// Adam optimizer parameters
	vector<float*> mt_weights;
    vector<float*> mt_biases;
	vector<float*> vt_weights;
    vector<float*> vt_biases;

public:

    // Construct the class
    SimpleNN(float* _X, float* _Y,
			 int _n, int _p, int _q,
			 const vector<int>& _layers) : n{_n}, p{_p}, q{_q}, layers{_layers}
	{
		// Copy Data
		cudaMallocManaged(&X, n*p*sizeof(float)); // n x p
		cudaMallocManaged(&Y, n*q*sizeof(float)); // n x q

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < p; ++j)
				X[idx(i,j,n)] = _X[idx(i,j,n)];
			for (int k = 0; k < q; ++k)
				Y[idx(i,k,n)] = _Y[idx(i,k,n)];
		}

		// initialize weights and biases
		num_layers = layers.size();

		for (int l = 0; l < num_layers-1; ++l)
		{
			float* W; cudaMallocManaged(&W, layers[l]*layers[l+1]*sizeof(float));
			float* b; cudaMallocManaged(&b, 1*layers[l+1]*sizeof(float));
			xavier_init(layers[l],layers[l+1],W,b);
			weights.push_back(W);
			biases.push_back(b);
		}

		// initialize hidden units
		for (int l = 0; l < num_layers; ++l)
		{
			float* H; cudaMallocManaged(&H, n*layers[l]*sizeof(float)); // n x layers[l]
			float* A; cudaMallocManaged(&A, n*layers[l]*sizeof(float)); // n x layers[l]
			H_list.push_back(H);
			A_list.push_back(A);
		}

		// initialize backpropagation units
		for (int l = 0; l < num_layers-1; ++l)
		{
			float* G; cudaMallocManaged(&G, n*layers[l+1]*sizeof(float)); // n x layers[l+1]
			float* loss_W; cudaMallocManaged(&loss_W, layers[l]*layers[l+1]*sizeof(float)); // layers[l] x layers[l+1]
			float* loss_b; cudaMallocManaged(&loss_b, 1*layers[l+1]*sizeof(float)); // 1 x layers[l+1]
			G_list.push_back(G);
			loss_weights.push_back(loss_W);
			loss_biases.push_back(loss_b);
		}

		// initialize adam prameters
		for (int l = 0; l < num_layers-1; ++l)
		{
			float* mt_W; cudaMallocManaged(&mt_W, layers[l]*layers[l+1]*sizeof(float));
			float* mt_b; cudaMallocManaged(&mt_b, 1*layers[l+1]*sizeof(float));
			float* vt_W; cudaMallocManaged(&vt_W, layers[l]*layers[l+1]*sizeof(float));
			float* vt_b; cudaMallocManaged(&vt_b, 1*layers[l+1]*sizeof(float));
			mt_weights.push_back(mt_W);
			mt_biases.push_back(mt_b);
			vt_weights.push_back(vt_W);
			vt_biases.push_back(vt_b);
		}
	}

	// destructor
	~SimpleNN()
	{
		cudaFree(X);
		cudaFree(Y);
		
		int num_layers = layers.size();

		for (int l = 0; l < num_layers-1; ++l)
		{
			cudaFree(weights[l]);
			cudaFree(biases[l]);
		}

		for (int l = 0; l < num_layers; ++l)
		{
			cudaFree(H_list[l]);
			cudaFree(A_list[l]);
		}

		for (int l = 0; l < num_layers-1; ++l)
		{
			cudaFree(G_list[l]);
			cudaFree(loss_weights[l]);
			cudaFree(loss_biases[l]);
		}

		for (int l = 0; l < num_layers-1; ++l)
		{
			cudaFree(mt_weights[l]);
			cudaFree(mt_biases[l]);
			cudaFree(vt_weights[l]);
			cudaFree(vt_biases[l]);
		}
	}

	// Loss Function
    void loss_function()
	{        
		// feed forward
        
		// H_list[0] = X; A_list[0] = X;
		copy<<<n,p>>>(X,H_list[0],A_list[0]);
		//cudaDeviceSynchronize();
	
        for (int l = 0; l < num_layers-1; ++l)
		{
			dense<<<n,layers[l+1]>>>(A_list[l], weights[l], biases[l], H_list[l+1], A_list[l+1], layers[l]);
			//cudaDeviceSynchronize();
		}

        // backpropagation
  		
		sub<<<n,layers[num_layers-1]>>>(H_list[num_layers-1],Y,G_list[num_layers-2]);
		//cudaDeviceSynchronize();
      
		for (int l = num_layers-2; l > 0; --l)
        {

			matmul1<<<layers[l], layers[l+1]>>>(H_list[l],G_list[l],loss_weights[l],n);
			//cudaDeviceSynchronize();

			row_sum<<<1, layers[l+1]>>>(G_list[l],loss_biases[l],n);
            //cudaDeviceSynchronize();

			backprop<<<n, layers[l]>>>(A_list[l],G_list[l],weights[l],G_list[l-1],layers[l+1]);
			//cudaDeviceSynchronize();

        }

        matmul1<<<layers[0], layers[1]>>>(H_list[0],G_list[0],loss_weights[0],n);
		//cudaDeviceSynchronize();

		row_sum<<<1, layers[1]>>>(G_list[0],loss_biases[0],n);
        //cudaDeviceSynchronize();
	}

	// training
	void train(int max_iter, float learning_rate)
	{
        auto start_time = high_resolution_clock::now();
		auto finish_time = high_resolution_clock::now();
		chrono::duration<float> elapsed;
		for (int it = 1; it < max_iter+1; ++it)
        {
            // Compute loss and gradients
			loss_function();
            
            // Update parameters
			for (int l = 0; l < num_layers-1; ++l)
			{

                stochastic_update_Adam<<<layers[l], layers[l+1]>>>(weights[l],
                                             					   loss_weights[l],
                                             					   mt_weights[l],
                                             					   vt_weights[l],
                                             					   learning_rate,
											 					   it);
				//cudaDeviceSynchronize();

                
                stochastic_update_Adam<<<1, layers[l+1]>>>(biases[l],
                                             			   loss_biases[l],
                                             			   mt_biases[l],
                                             			   vt_biases[l],
                                             			   learning_rate,
											 			   it);
				//cudaDeviceSynchronize();
            }

			cudaDeviceSynchronize();
    
            if (it % 10 == 0)
			{
				float loss = 0.5*squared_l2_error(H_list[num_layers-1],Y,n,q);

				finish_time = high_resolution_clock::now();
				elapsed = finish_time-start_time;
                cout << "iteration: " << it 
					 << ", loss: " << loss 
					 << ", time: " << elapsed.count() 
					 << ", learning rate: " << learning_rate << "\n";
                start_time = high_resolution_clock::now();
			}
		}
	}

	// predictions
	void test(float* X_star, int n_star, float* Y_pred)
	{
     	vector<float*> H_star_list; // hidden units
    	vector<float*> A_star_list; // activated units

        // initialize hidden units
		for (int l = 0; l < num_layers; ++l)
		{
			float* H_star; cudaMallocManaged(&H_star, n_star*layers[l]*sizeof(float)); // n_star x layers[l]
			float* A_star; cudaMallocManaged(&A_star, n_star*layers[l]*sizeof(float)); // n_star x layers[l]
			H_star_list.push_back(H_star);
			A_star_list.push_back(A_star);
		}
        
		copy<<<n_star,p>>>(X_star,H_star_list[0],A_star_list[0]);
		//cudaDeviceSynchronize();

        for (int l = 0; l < num_layers-1; ++l)
		{
			dense<<<n_star,layers[l+1]>>>(A_star_list[l],
									 weights[l], biases[l],
									 H_star_list[l+1],
									 A_star_list[l+1],
									 layers[l]);
			//cudaDeviceSynchronize();
		}
        
		cudaDeviceSynchronize();

		for (int i = 0; i < n_star; ++i)
			for (int k = 0; k < q; ++k)
				Y_pred[idx(i,k,n)] = (H_star_list[num_layers-1])[idx(i,k,n)];

		for (int l = 0; l < num_layers; ++l)
		{
			cudaFree(H_star_list[l]);
			cudaFree(A_star_list[l]);
		}

	}


};

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 4  ////////////////////////////////////
///////////////////// Main Function /////////////////////////////////
/////////////////////////////////////////////////////////////////////


int main()
{

	// load training data
	auto training_data = load_data("./training_data.csv");
	int N_train = get<2>(training_data);
	vector<float> X_training_data = get<0>(training_data);
	vector<float> Y_training_data = get<1>(training_data);

	float* X_train; cudaMallocManaged(&X_train, N_train*1*sizeof(float)); // N x 1
	float* Y_train; cudaMallocManaged(&Y_train, N_train*1*sizeof(float)); // N x 1

	for (int i = 0; i < N_train ; ++i)
	{
		X_train[i] = X_training_data[i];
		Y_train[i] = Y_training_data[i];
	}

	// load test data
	auto test_data = load_data("./test_data.csv");
	int N_test = get<2>(test_data);
	vector<float> X_test_data = get<0>(test_data);
	vector<float> Y_test_data = get<1>(test_data);

	float* X_test; cudaMallocManaged(&X_test, N_test*1*sizeof(float)); // N_star x 1
	float* Y_test; cudaMallocManaged(&Y_test, N_test*1*sizeof(float)); // N_star x 1

	for (int i = 0; i < N_test ; ++i)
	{
		X_test[i] = X_test_data[i];
		Y_test[i] = Y_test_data[i];
	}

	// build the model
	SimpleNN model(X_train, Y_train,
				   N_train, 1, 1,
				   {1,10,10,10,10,10,10,10,10,10,10,1});

	// train the model
	model.train(10000,1e-3);
	model.train(10000,1e-4);
	model.train(10000,1e-5);
	model.train(10000,1e-6);

	// test the model
	float* Y_pred; cudaMallocManaged(&Y_pred, N_test*1*sizeof(float)); // N_star x 1
	model.test(X_test, N_test, Y_pred);

	//compute relative L2 error
	cout << "rel. L2 error:" << relative_l2_error(Y_pred, Y_test, N_test, 1) << "\n";

	// plot results
	/*
	for (int i = 0; i < N_test ; ++i)
	{
		cout << X_test[i] << " " << Y_test[i] << " " << Y_pred[i] << "\n";
	}
	*/

	cudaFree(X_train);
	cudaFree(Y_train);
	cudaFree(X_test);
	cudaFree(Y_test);
	cudaFree(Y_pred);

	return 0;
}
