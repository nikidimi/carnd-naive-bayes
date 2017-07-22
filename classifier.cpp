#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
  for(int i = 0; i < data.size(); i++) {
    if (labels[i] == "left") {
      left.add_data(data[i]);
    }
    if (labels[i] == "keep") {
      keep.add_data(data[i]);
    }
    if (labels[i] == "right") {
      right.add_data(data[i]);
    }
  }

  left.fit(data.size());
  keep.fit(data.size());
  right.fit(data.size());
}

string GNB::predict(vector<double> input)
{
  double l = left.predict(input);
  double k = keep.predict(input);
  double r = right.predict(input);

  if (l < k && l < r) {
    return this->possible_labels[0];
  }
  if (k < l && k < r) {
    return this->possible_labels[1];
  }
  if (r < k && r < l) {
    return this->possible_labels[2];
  }

	return this->possible_labels[1];

}

void GNB_Class::add_data(vector<double> input) {
  s.add_data(input[0]);
  d.add_data(input[1]);
  s_dot.add_data(input[2]);
  d_dot.add_data(input[3]);
}

void GNB_Class::fit(int total_count) {
  s.fit();
  d.fit();
  s_dot.fit();
  d_dot.fit();

  class_probabilty = s.data.size() / (double) total_count;
}

double GNB_Class::predict(vector<double> input) {
  double s_prob = s.get_probability(input[0]);
  double d_prob = d.get_probability(input[1]);
  double s_dot_prob = s_dot.get_probability(input[2]);
  double d_dot_prob = d_dot.get_probability(input[3]);

  //cout << "s_prob=" << s_prob << " d_prob=" << d_prob;
  //cout << " s_dot_prob=" << s_dot_prob << " d_dot_prob=" << d_dot_prob << endl;

  return s_prob *  d_prob * d_prob * d_dot_prob;
}

void GNB_Feature::add_data(double input) {
  data.push_back(input);
}

double GNB_Feature::calc_mean() {
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    sum += data[i];
  }
  mean = sum/data.size();
}

double GNB_Feature::calc_variance() {
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    double val = (data[i] - mean);
    sum += val * val;
  }
  variance_squared = sum/data.size();
}

void GNB_Feature::fit() {
  calc_mean();
  calc_variance();

  cout << mean << " "  << variance_squared << endl;
}

double GNB_Feature::get_probability(double x) {
  if (variance_squared == 0) {
    return 1;
  }
  double power = -(x - mean) * (x - mean) / (2 * variance_squared);
  double a = sqrt(2 * M_PI * variance_squared);
  return 1/(a * pow(M_E, power));
}
