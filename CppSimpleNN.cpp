/*********************
@author: Maziar Raissi
*********************/

/**************************************************
To compile and run use:
    g++ -std=c++11 CppSimpleNN.cpp -o CppSimpleNN
    ./CppSimpleNN
**************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <array>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 1  ////////////////////////////////////
//////////////////// Matrix Algebra /////////////////////////////////
/////////////////////////////////////////////////////////////////////

constexpr int idx(int _i, int _j, int _m) { return _i + _j*_m; }

// Matrix Class
class Matrix
{

private:
	
	double* mat;
	int m;
	int n;

public:
	
	// default constructor
	Matrix(): mat{nullptr}, m{0}, n{0} {};

	// ordinary constructor
	Matrix(int _m, int _n) : mat{new double[_m*_n]}, m{_m}, n{_n}
	{
    	for (int idx = 0; idx < m*n; ++idx)
			mat[idx] = 0.0;
	}

	// copy constructor
	Matrix(const Matrix& rhs) : mat{new double[rhs.m*rhs.n]}, m{rhs.m}, n{rhs.n}
	{
    	for (int idx = 0; idx < m*n; ++idx)
			mat[idx] = rhs.mat[idx];

		// cout << "copy constructor" << "\n";
	}

	// move constructor
	Matrix(Matrix&& rhs) : mat{rhs.mat}, m{rhs.m}, n{rhs.n}
	{
		rhs.mat = nullptr;
		rhs.m = 0;
		rhs.n = 0;

		// cout << "move constructor" << "\n";
	}

	// copy assignment
	Matrix& operator=(const Matrix& rhs)
	{
		double* p;
		p = new double[rhs.m*rhs.n];
		
		for (int idx = 0; idx < rhs.m*rhs.n; ++idx)
			p[idx] = rhs.mat[idx];
		
		delete[] mat;
		mat = p;		

		m = rhs.m;
		n = rhs.n;

		// cout << "copy assignment" << "\n";

		return *this;
	}


	// move assignment
	Matrix& operator=(Matrix&& rhs)
	{
		mat = rhs.mat;
		m = rhs.m;
		n = rhs.n;

		rhs.mat = nullptr;
		rhs.m = 0;
		rhs.n = 0;

		// cout << "move assignment" << "\n";

		return *this;
	}


	// destructor
	~Matrix()
	{
		delete[] mat;
	};

	// element access
	double& operator()(int i, int j)
	{
		return mat[idx(i,j,m)];
	}

	const double& operator()(int i, int j) const
	{
		return mat[idx(i,j,m)];
	}

	int num_rows() const
	{
		return m;
	}

	int num_cols() const
	{
		return n;
	}

	// print matrix
	void Print()
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
};

// matrix addition
Matrix operator+(const Matrix& lhs, const Matrix &rhs)
{
	Matrix result(lhs.num_rows(),lhs.num_cols());
	
	// Broadcast
	if (rhs.num_rows() == 1)
	{
		for (int i = 0; i < lhs.num_rows(); ++i)
			for (int j = 0; j < lhs.num_cols(); ++j)
				result(i,j) = lhs(i,j) + rhs(0,j);
		
		return result;	
	}
	
	for (int i = 0; i < lhs.num_rows(); ++i)
		for (int j = 0; j < lhs.num_cols(); ++j)
			result(i,j) = lhs(i,j) + rhs(i,j);

	return result;
}

// matrix subtraction
Matrix operator-(const Matrix& lhs, const Matrix& rhs)
{
	Matrix result(lhs.num_rows(),lhs.num_cols());
	
	for (int i = 0; i < lhs.num_rows(); ++i)
		for (int j = 0; j < lhs.num_cols(); ++j)
			result(i,j) = lhs(i,j) - rhs(i,j);

	return result;
}

// scalar addition
Matrix operator+(const double& lhs, const Matrix& rhs)
{
	Matrix result(rhs.num_rows(),rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = lhs + rhs(i,j);

	return result;
}

// scalar subtraction
Matrix operator-(const double& lhs, const Matrix& rhs)
{
	Matrix result(rhs.num_rows(),rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = lhs - rhs(i,j);

	return result;
}

// matrix multiplication (pointwise)
Matrix operator*(const Matrix& lhs, const Matrix& rhs)
{
	Matrix result(lhs.num_rows(),lhs.num_cols());
	
	for (int i = 0; i < lhs.num_rows(); ++i)
		for (int j = 0; j < lhs.num_cols(); ++j)
			result(i,j) = lhs(i,j) * rhs(i,j);

	return result;
}

// scaler multiplication
Matrix operator*(const double& lhs, const Matrix& rhs)
{
	Matrix result(rhs.num_rows(),rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = lhs * rhs(i,j);

	return result;
}

// matrix multiplication
Matrix matmul(const Matrix& lhs, const Matrix& rhs, bool transpose_lhs = false, bool transpose_rhs = false)
{

	if (!transpose_lhs && !transpose_rhs)
	{
		// cout << 1 << " false" << " false" << "\n";

		Matrix result(lhs.num_rows(),rhs.num_cols());
	
		for (int i = 0; i < lhs.num_rows(); ++i)
			for (int j = 0; j < rhs.num_cols(); ++j)
			{
				result(i,j) = 0;
				for (int k = 0; k < lhs.num_cols(); ++k)
					result(i,j) += lhs(i,k) * rhs(k,j);
			}

		return result;
	}

	if (!transpose_lhs && transpose_rhs)
	{
		// cout << 2 << " false" << " true" << "\n";

		Matrix result(lhs.num_rows(),rhs.num_rows());
	
		for (int i = 0; i < lhs.num_rows(); ++i)
			for (int j = 0; j < rhs.num_rows(); ++j)
			{
				result(i,j) = 0;
				for (int k = 0; k < lhs.num_cols(); ++k)
					result(i,j) += lhs(i,k) * rhs(j,k);
			}

		return result;
	}

	if (transpose_lhs && !transpose_rhs)
	{

		// cout << 3 << " true" << " false" << "\n";

		Matrix result(lhs.num_cols(),rhs.num_cols());
	
		for (int i = 0; i < lhs.num_cols(); ++i)
			for (int j = 0; j < rhs.num_cols(); ++j)
			{
				result(i,j) = 0;
				for (int k = 0; k < lhs.num_rows(); ++k)
					result(i,j) += lhs(k,i) * rhs(k,j);
			}

		return result;
	}
	
	// cout << 4 << " true" << " true" << "\n";

	Matrix result(lhs.num_cols(),rhs.num_rows());

	for (int i = 0; i < lhs.num_cols(); ++i)
		for (int j = 0; j < rhs.num_rows(); ++j)
		{
			result(i,j) = 0;
			for (int k = 0; k < lhs.num_rows(); ++k)
				result(i,j) += lhs(k,i) * rhs(j,k);
		}

	return result;
}

// scalar division
Matrix operator/(const double& lhs, const Matrix& rhs)
{
	Matrix result(rhs.num_rows(),rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = lhs / rhs(i,j);

	return result;
}

// squared sum
double squared_sum(const Matrix& input)
{
	double sum = 0;
	for (int i = 0; i < input.num_rows(); ++i)
		for (int j = 0; j < input.num_cols(); ++j)
			sum += input(i,j) * input(i,j);

	return sum;
}

// sum rows
Matrix row_sum(const Matrix& rhs)
{
	Matrix sum(1,rhs.num_cols());
	
	for (int j = 0; j < rhs.num_cols(); ++j)
		for (int i = 0; i < rhs.num_rows(); ++i)
			sum(0,j) += rhs(i,j);

	return sum;
}

// pointwise tanh
Matrix tanh(const Matrix& rhs)
{
	Matrix result(rhs.num_rows(), rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = tanh(rhs(i,j));

	return result;
}

// pointwise sqrt
Matrix sqrt(const Matrix& rhs)
{
	Matrix result(rhs.num_rows(), rhs.num_cols());
	
	for (int i = 0; i < rhs.num_rows(); ++i)
		for (int j = 0; j < rhs.num_cols(); ++j)
			result(i,j) = sqrt(rhs(i,j));

	return result;
}

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 2  ////////////////////////////////////
/////////////////// Helper Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////

// Function to Load Data
tuple< Matrix, Matrix, int > load_data(string file_name)
{
	ifstream ifs {file_name};
	if (!ifs)
		cerr << "Could not open file:" << file_name << "\n";

	vector<double> X;
	vector<double> Y;
	string line;
	while (getline(ifs, line))
	{
		double x, y;
		istringstream line_stream(line);
		line_stream >> x >> y;
		X.push_back(x);
		Y.push_back(y);
	}
	
	int N = X.size();

	Matrix X_mat(N,1);
	Matrix Y_mat(N,1);

	for (int i = 0; i < N; ++i)
	{
		X_mat(i,0) = X[i];
		Y_mat(i,0) = Y[i];
	}

	return make_tuple(X_mat, Y_mat, N);
}

// Xavier Initializer
Matrix xavier_init(int in_dim, int out_dim)
{
    double xavier_stddev = sqrt(2.0/(in_dim + out_dim));

	int seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
    normal_distribution<double> distribution(0.0,xavier_stddev);

    Matrix W(in_dim, out_dim);
    
	for (int i = 0; i < in_dim; ++i)
		for (int j = 0; j < out_dim; ++j)
			W(i,j) = distribution(generator);

    return W;
}

// Initialize Neural Network (weights & biases)
tuple< vector<Matrix>, vector<Matrix> > initialize_NN(const vector<int>& layers)
{
	vector<Matrix> weights;
    vector<Matrix> biases;
    int num_layers = layers.size();

	for (int l = 0; l < num_layers-1; ++l)
	{
		Matrix W = xavier_init(layers[l],layers[l+1]);
		Matrix b({1,layers[l+1]});
		weights.push_back(W);
		biases.push_back(b);
	}

	return make_tuple(weights, biases);
}

// Adam Optimizer
tuple< Matrix, Matrix, Matrix > stochastic_update_Adam(Matrix& w,
													   Matrix& grad_w,
													   Matrix& mt,
													   Matrix& vt,
													   double lrate,
													   int iteration)
{
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    mt = beta1*mt + (1.0-beta1)*grad_w;
    vt = beta2*vt + (1.0-beta2)*(grad_w*grad_w);

    Matrix mt_hat = (1.0/(1.0-pow(beta1,iteration)))*mt;
    Matrix vt_hat = (1.0/(1.0-pow(beta2,iteration)))*vt;

    Matrix scal = 1.0/(epsilon + sqrt(vt_hat));

    w = w - (lrate*(mt_hat*scal));
    
    return make_tuple(w, mt, vt);
}

/////////////////////////////////////////////////////////////////////
//////////////////////// Part 3  ////////////////////////////////////
//////////////////// Neural Networks ////////////////////////////////
/////////////////////////////////////////////////////////////////////

// Neural Networks Class [ y = f(x) ]
class SimpleNN
{

private:

	Matrix X;
	Matrix Y;

	vector<int> layers;
	int num_layers;

	vector<Matrix> weights;
	vector<Matrix> biases;

	vector<Matrix> H_list; // hidden units
	vector<Matrix> A_list; // activated units

	vector<Matrix> G_list; // backpropagation units
	vector<Matrix> loss_weights; // gradients of the loss with respect to weights
	vector<Matrix> loss_biases; // gradients of the loss with respect to biases

	// Adam optimizer parameters
	vector<Matrix> mt_weights;
	vector<Matrix> mt_biases;
	vector<Matrix> vt_weights;
	vector<Matrix> vt_biases;

public:

    // Construct the class
    SimpleNN(const Matrix& _X, const Matrix& _Y, const vector<int>& _layers) : X{_X}, Y{_Y}, layers{_layers}
	{

		int n = X.num_rows();
		num_layers = layers.size();

		// initialize weights and biases
        auto out = initialize_NN(_layers);
		weights = get<0>(out);
		biases = get<1>(out);

		// initialize hidden units
		for (int l = 0; l < num_layers; ++l)
		{
			Matrix H(n,layers[l]); // n x layers[l]
			Matrix A(n,layers[l]); // n x layers[l]
			H_list.push_back(H);
			A_list.push_back(A);
		}

		// initialize backpropagation units
		for (int l = 0; l < num_layers-1; ++l)
		{
			Matrix G(n,layers[l+1]); // n x layers[l+1]
			Matrix loss_W(layers[l],layers[l+1]); // layers[l] x layers[l+1]
			Matrix loss_b(1,layers[l+1]); // 1 x layers[l+1]
			G_list.push_back(G);
			loss_weights.push_back(loss_W);
			loss_biases.push_back(loss_b);
		}
		
		// initialize adam prameters
		for (int l = 0; l < num_layers-1; ++l)
		{
			Matrix mt_W(layers[l],layers[l+1]);
			Matrix mt_b(1,layers[l+1]);
			Matrix vt_W(layers[l],layers[l+1]);
			Matrix vt_b(1,layers[l+1]);
			mt_weights.push_back(mt_W);
			mt_biases.push_back(mt_b);
			vt_weights.push_back(vt_W);
			vt_biases.push_back(vt_b);
		}

	}

	// Loss Function
    double loss_function()
	{   
		// feed forward
        
        H_list[0] = X;
        A_list[0] = X;
        
        for (int l = 0; l < num_layers-1; ++l)
		{
            H_list[l+1] = matmul(A_list[l],weights[l]) + biases[l];
            A_list[l+1] = tanh(H_list[l+1]);
		}
        
        double loss = 0.5*squared_sum(H_list[num_layers-1]-Y);

        // backpropagation
                
        G_list[num_layers-2] = H_list[num_layers-1] - Y;
        
		for (int l = num_layers-2; l > 0; --l)
        {    
            loss_weights[l] = matmul(H_list[l],G_list[l],true,false);
            loss_biases[l] = row_sum(G_list[l]);
            
            G_list[l-1] = (1.0 - A_list[l]*A_list[l]) * matmul(G_list[l], weights[l],false,true);
        }

        loss_weights[0] = matmul(H_list[0],G_list[0],true,false);
        loss_biases[0] = row_sum(G_list[0]);

        
        return loss;
	}

    void train(int max_iter, double learning_rate)
	{
        
		double loss;

        auto start_time = high_resolution_clock::now();
		auto finish_time = high_resolution_clock::now();
		chrono::duration<double> elapsed;
		for (int it = 1; it < max_iter+1; ++it)
        {
            // Compute loss and gradients
			loss = loss_function();
            
            // Update parameters
			for (int l = 0; l < num_layers-1; ++l)
			{

                auto adam1 = stochastic_update_Adam(weights[l],
                                               		loss_weights[l],
                                               		mt_weights[l],
                                               		vt_weights[l],
                                               		learning_rate,
											   		it);

				weights[l] = get<0>(adam1);
				mt_weights[l] = get<1>(adam1);
				vt_weights[l] = get<2>(adam1);
                
                auto adam2 = stochastic_update_Adam(biases[l],
                                               		loss_biases[l],
                                               		mt_biases[l],
                                               		vt_biases[l],
                                               		learning_rate,
											   		it);

				biases[l] = get<0>(adam2);
				mt_biases[l] = get<1>(adam2);
				vt_biases[l] = get<2>(adam2);
            }
    
            if (it % 10 == 0)
			{
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
	
	// Prediction
	Matrix test(Matrix& X_star)
	{
        int num_layers = layers.size();
        
        Matrix H = X_star;
        Matrix A = X_star;
        
        for (int l = 0; l < num_layers-1; ++l)
		{
            H = matmul(A,weights[l]) + biases[l];  
            A = tanh(H);
		}
        
        return H;
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
	Matrix X_train = get<0>(training_data);
	Matrix Y_train = get<1>(training_data);

	// plot traning data
	/*
	for (int i = 0; i < X_train.get_shape()[0] ; ++i)
		cout << X_train(i,0) << " " << Y_train(i,0) << "\n";
	*/

	// load test data
	auto test_data = load_data("./test_data.csv");
	int N_test = get<2>(test_data);
	Matrix X_test = get<0>(test_data);
	Matrix Y_test = get<1>(test_data);

	// plot test data
	/*
	for (int i = 0; i < X_test.get_shape()[0] ; ++i)
		cout << X_test(i,0) << " " << Y_test(i,0) << "\n";
	*/

	// build the model
	SimpleNN model(X_train, Y_train, {1,10,10,10,10,10,10,10,10,10,10,1});

	// train the model
	model.train(10000,1e-3);
	model.train(10000,1e-4);
	model.train(10000,1e-5);
	model.train(10000,1e-6);

	// test the model
	Matrix Y_pred = model.test(X_test);

	// plot results
	/*
	for (int i = 0; i < X_test.get_shape()[0] ; ++i)
		cout << X_test(i,0) << " " << Y_test(i,0) << " " << Y_pred(i,0) << "\n";
	*/

	cout <<	"relative L-2 error: " << std::sqrt(squared_sum(Y_test - Y_pred)/squared_sum(Y_test)) << "\n";

	return 0;
}
