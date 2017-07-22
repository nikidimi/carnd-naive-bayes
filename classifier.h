#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB_Feature {
public:
  vector<double> data;
  double mean;
  double variance_squared;

  void add_data(double input);
  double calc_mean();
  double calc_variance();
  void fit();

  double get_probability(double x);
};

class GNB_Class {
public:
  GNB_Feature s;
  GNB_Feature d;
  GNB_Feature s_dot;
  GNB_Feature d_dot;

  double class_probabilty;

  void add_data(vector<double> input);
  void fit(int total_count);
  double predict(vector<double>);
};

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  string predict(vector<double>);

private:
  GNB_Class left;
  GNB_Class keep;
  GNB_Class right;
};

#endif
