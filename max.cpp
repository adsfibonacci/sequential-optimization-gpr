#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <eigen3/Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <numbers>
#include <random>
#include <utility>

using namespace std;




float euc(const vector<float>& p1, const vector<float>& p2) {
  vector<vector<float>> p2_tiled(pow(3, p2.size()));
  float sum = numeric_limits<float>::infinity();

  for(unsigned int i = 0; i < p2_tiled.size(); ++i) {
	vector<float> new_point = p2;
	int temp = i;
	for(unsigned int j = 0; j < p2.size(); ++j) {
	  int shift = (temp % 3) - 1;
	  temp /= 3;
	  new_point[j] = p2[j] + (float)shift;
	}
	p2_tiled[i] = new_point;
  }
  
  for(unsigned int i = 0; i < p2_tiled.size(); ++i) {
	vector<float> d;
	d.reserve(p1.size());
	for(unsigned int j = 0; j < p1.size(); ++j) {
	  d.push_back(pow(p1[j] - p2_tiled[i][j], 2));	  
	}
	float t = reduce(d.begin(), d.end());
	if(t < sum) sum = t;
  }
  
  return pow(sum, .5);
}
tuple<float, unsigned int, unsigned int> min_location(const vector<vector<float>>& points) {
  tuple<float, unsigned int, unsigned int> run_min = {numeric_limits<float>::infinity(), 0, 0};
  for(unsigned int i = 0; i < points.size(); ++i) {
	for(unsigned int j = 0; j < i; ++j) {
	  float dis = euc(points[i], points[j]);
	  if(dis < get<0>(run_min)) run_min = {dis, i, j};	  
	}
  }
  return run_min;
}
float min_val(const vector<vector<float>>& points) {
  float run_min = numeric_limits<float>::infinity();
  for(unsigned int i = 0; i < points.size(); ++i) {
	for(unsigned int j = 0; j < i; ++j) {
	  float dis = euc(points[i], points[j]);
	  if(dis < run_min) run_min = dis;	  
	}
  }
  return run_min;
}
vector<vector<float>> max_point_movement(vector<vector<float>> points, unsigned int index) {

  unsigned int max_coordinate = 0;
  float max_min = -numeric_limits<float>::infinity();
  float max_offset = 0;

  for(unsigned int i = 0; i < points[index].size(); ++i) {

	float measure = min_val(points);
	float original = points[index][i];
	float pos = measure;
	float neg = measure;
	float offset = .001;
	
	float round;
	
	do {
	  points[index][i] = fmod(points[index][i] + offset, 1.0);
	  pos = measure;
	  measure = min_val(points);
	} while(measure > pos);
	round = points[index][i] - original;
	do {
	  points[index][i] = fmod(points[index][i] - offset + 1, 1.0);
	  neg = measure;
	  measure = min_val(points);
	} while(measure > neg);
	points[index][i] += offset;

	if(measure > max_min) {
	  max_coordinate = i;
	  max_min = measure;
	  max_offset = points[index][i] - original;
	}

	points[index][i] = original;
  }
  points[index][max_coordinate] = fmod(points[index][max_coordinate] + max_offset, 1.0);
  return points;
}
vector<vector<float>> maximin(unsigned int count, unsigned int dim) {
  float max_min = -numeric_limits<float>::infinity();
  float tested = 0;
  
  vector<vector<float>> data;
  data.push_back(vector<float>(dim, .95));
  data.push_back(vector<float>(dim, .05));
  
  tuple<float, unsigned int, unsigned int> temp = min_location(data);
  unsigned int movement_index;

  data = max_point_movement(data, data.size() - 1);

  for(unsigned int i = 2; i < count; ++i) {
	data.push_back(vector<float>(dim, .95));
	temp = min_location(data);
	tested = min_val(data);
	temp = min_location(data);
  	while( fabs(max_min - tested) > pow(10, -4)) {
	  data = max_point_movement(data, data.size() - 1);
	  data = max_point_movement(data, get<1>(temp));
	  data = max_point_movement(data, get<2>(temp));	  
  	  max_min = tested;
  	  tested = min_val(data);
  	  temp = min_location(data);
  	}
  }
  return data;
}
float pdf(float x, float m, float std) {
  return pow(M_E, -pow(x-m, 2) / (2*pow(std, 2))) / ( std * pow(2 * M_PI, .5));
}
float cdf(float x, float m, float std) {
  return .5 * erfc( -(x-m) / (std * pow(2, .5)));
}
float func(float x, float y) {
  return (4 - 2.1 * pow(x,2) + pow(x, 4/3))*pow(x,2) + x*y + (-4 + 4*pow(y,2))*pow(y,2);
}
float scale(float x) {
  return 4.1*x-2.1;
}
float scale_inv(float x) {
  return (x + 2.1) / 4.1;
}
float f(float x, float y) {
  return func(scale(x), y);
}
pair<vector<float>, float> brute_force_max() {
  float offset = .0001;
  vector<float> curr_max;
  float brute_max = -numeric_limits<float>::infinity();
  for(float i = 0; i < 1; i+= offset) {
	for(float j = 0; j < 1; j += offset) {	  
	  float ev = f(i, j);
	  if(ev > brute_max) {
		brute_max = ev;
		curr_max = {i, j};
	  }
	}
  }
  return make_pair(curr_max, brute_max);
}
using std::cout, std::endl, std::pair, std::function, std::make_pair, std::bind, std::vector;
using Eigen::Dynamic, Eigen::Matrix, Eigen::Map;
float gaussian_kernel(const unsigned int dim,
					  const float *d1,
					  const float *d2,
					  const float sigma = 1,
					  const float length = 1) {
  float totals = 0;
  for(unsigned int i = 0; i < dim; ++i) {
	totals += pow(*(d1 + i) - *(d2 + i), 2);
  }
  return pow(sigma, 2) * exp(-totals / ( 2*pow(length,2)));
}
typedef function<float(const float*, const float*)> KernType;
KernType kern(unsigned int dim, float std, float length) {
  return bind(gaussian_kernel, dim, std::placeholders::_1, std::placeholders::_2, std, length);
}
Matrix<float, Dynamic, Dynamic> covariance_matrix(
												  const unsigned int dim,
												  const unsigned int count,
												  const float *data,
												  const KernType ker
												  ) {
  Matrix<float, Dynamic, Dynamic> cov_mat(count, count);
  for(unsigned int i = 0; i < count; ++i) {
  	for(unsigned int j = 0; j <= i; ++j) {
  	  cov_mat(i, j) = ker(data + i * dim, data + j * dim);
  	  cov_mat(j, i) = cov_mat(i, j);
  	}
  }
  return cov_mat;
}
Matrix<float, Dynamic, Dynamic> covariance_matrix(
												  const vector<vector<float>>& data,
												  const KernType ker
												  ) {
  Matrix<float, Dynamic, Dynamic> cov_mat(data.size(), data.size());
  for(unsigned int i = 0; i < data.size(); ++i) {
  	for(unsigned int j = 0; j <= i; ++j) {
  	  cov_mat(i, j) = ker(&data[i][0], &data[j][0]);
  	  cov_mat(j, i) = cov_mat(i, j);
  	}
  }
  return cov_mat;
}
Matrix<float, Dynamic, 1> covariance_vector(
											const unsigned int dim,
											const unsigned int count,
											const float *data,
											const float *point,
											const KernType ker
											) {
  Matrix<float, Dynamic, 1> cov_vec(count);
  for(unsigned int i = 0; i < count; ++i) {
	cov_vec(i) = ker(data + i * dim, point);
  }
  return cov_vec;
}
Matrix<float, Dynamic, 1> covariance_vector(
											const vector<vector<float>>& data,
											const vector<float>& point,
											const KernType ker
											) {
  Matrix<float, Dynamic, 1> cov_vec(data.size());
  for(unsigned int i = 0; i < data.size(); ++i) {
	cov_vec(i) = ker(&data[i][0], &point[0]);
  }
  return cov_vec;
}
pair<float, float> predict(
						   const Matrix<float, Dynamic, 1>& response,
						   const Matrix<float, Dynamic, Dynamic>& cov_mat,
						   const Matrix<float, Dynamic, 1>& cov_vec
						   ) {
  return make_pair(cov_vec.transpose() * cov_mat.inverse() * response,
						1 - cov_vec.transpose() * cov_mat.inverse() * cov_vec);
}
pair<float, float> gpr_model(const unsigned int dim, const unsigned int count,
							 const float *data,
							 const float *response,
							 const float *point,
							 KernType ker) {
  Matrix<float, Dynamic, Dynamic> cov_mat(count, count);
  Matrix<float, Dynamic, 1> cov_vec(count);
  Map<const Matrix<float, Dynamic, 1>> res(response, count);
  cov_mat = covariance_matrix(dim, count, data, ker);
  cov_vec = covariance_vector(dim, count, data, point, ker);
  return predict(res, cov_mat, cov_vec);
}
pair<float, float> gpr_model(
							 const vector<vector<float>>& data,
							 const vector<float>& response,
							 const vector<float>& point,
							 KernType ker
							 ) {
  Matrix<float, Dynamic, Dynamic> cov_mat(data.size(), data.size());
  Matrix<float, Dynamic, 1> cov_vec(data.size());
  Map<const Matrix<float, Dynamic, 1>> res(&response[0], response.size());
  cov_mat = covariance_matrix(data, ker);
  cov_vec = covariance_vector(data, point, ker);
  return predict(res, cov_mat, cov_vec);
}
float ei(const vector<vector<float>>& data,
		 const vector<float>& response,
		 const vector<float> &test,
		 KernType ker) {
 
  unsigned int count = data.size();
  unsigned int dim = data[0].size();
  float best = *max_element(test.begin(), test.end());
  pair<float, float> mv = gpr_model(data, response, test, ker);  
  mv.second = pow(mv.second, .5);

  float z = max(mv.first - best, 0.0f);
  return z * cdf(z/mv.second, 0, 1) + mv.second * pdf(z/mv.second, 0, 1);
}
vector<float> argmax(const vector<vector<float>>& data,
					 const vector<float>& response,
					 KernType ker) {
  
  vector<float> curr_max(data[0].size());
  float best_ei = -numeric_limits<float>::infinity();

  float offset = .01;
  for(float i = 0.0; i < 1.0; i += offset) {
	for(float j = 0.0; j < 1.0; j += offset) {
	  vector<float> test = {i, j};
	  float test_ei = ei(data, response, test, ker);
	  if(test_ei > best_ei) {
		best_ei = test_ei;
		curr_max = test;
	  }
	}
  }
  return curr_max;
}

pair<vector<float>, float> seq_max(unsigned int budget, unsigned int given,
						   KernType ker) {
  
  vector<float> arg_max;
  float seq;

  unsigned int dim = 2;

  vector<vector<float>> data = maximin(given, dim);
  vector<float> response(given);
  cout << "Given: \n";
  for(unsigned int i = 0; i < given; ++i) {
	response[i] = f(data[i][0], data[i][1]);
	cout << "(" << data[i][0] << ", " << data[i][1] <<
	  ") \\mapsto " << response[i] << "\\newline" << endl;
  }  

  cout << "Predicted: \n";
  for(unsigned int i = given; i < budget; ++i) {
	vector<float> am = argmax(data, response, ker);
	data.push_back(am);
	response.push_back(f(am[0], am[1]));
	cout << "(" << data[i][0] << ", " << data[i][1] <<
	  ") \\mapsto " << response[i] << "\\newline" << endl;
  }
  vector<float>::iterator seq_max_p =
	max_element(response.begin(), response.end());
  
  arg_max = data[ (unsigned int)(response.end() - seq_max_p) - 2 ];
  seq = *seq_max_p;
  
  return make_pair(arg_max, seq);
}
int main() {
  pair<vector<float>, float> brute; // = brute_force_max();
  pair<vector<float>, float> estim;
  brute.first = {.809422, .99954};
  brute.second = 4.33683;

  unsigned int budget = 12;
  unsigned int count = 5;
  //for(unsigned int count = 3; count < budget; ++count) 
  {
	KernType ker = kern(2, 1, pow(2, -.5));
	estim = seq_max(budget, count, ker);
	cout << count << " predetermined,\t(" <<
	  estim.first[0] << ", " << estim.first[1] << ") \\mapsto "
		 << estim.second << " (sequential max)" << endl << endl;
  }
  cout << "The actual max is " << brute.second <<
	" as obtained from " << brute.first[0] <<
	", " << brute.first[1] << endl;
  }
