/*
 * random_forests.h
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#ifndef RANDOM_FORESTS_H_
#define RANDOM_FORESTS_H_
#include<vector>
#include"decision_tree.h"
#include"training_set.h"
#include"node.h"
#include<ctime>
#include<cstdlib>
#include<cstdio>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<iomanip>

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/foreach.hpp>
#else
#include<thread>
#include<mutex>
#include<random>
#include<future>
#include<chrono>
#endif
#endif

#include<exception>
#include<Rcpp.h>

using namespace std;

class RandomForests{
private:
	TrainingSet* training_set_;  // total training set
	int trees_num_;


	vector<vector<int> > bagging_set_;  // training set for each tree
	vector<vector<int> > OOBSet;
	vector<int> tree_seeds;  // store seeds for each tree

	double OOB_error_rate_;
	double strength_;
	double correlation_;
	double c_s2_;

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	boost::mutex mut;
#else
	mutex mut;
#endif
#endif

	static const string TREES;
	static const string SELECTED_STATUS;
	static const string ESTIMATION;
	static const string OOB_ERROR_RATE;
	static const string STRENGTH;
	static const string CORRELATION;
	static const string C_S2;
	static const string OOB_ERROR_RATES_FOR_EACH_TREE;


public:
	vector<DecisionTree*> random_forests_;
	vector<vector<bool> > selected_status_;  // <tree index> : <training set for the tree>

	RandomForests();
	RandomForests(SEXP& rf, bool isPart = false);
	
	RandomForests(TrainingSet* training_set,int tree_num, Rcpp::IntegerVector seeds);

	~RandomForests();

	vector<DecisionTree*> get_trees();
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
	void GenerateRF(int nvars,bool isWeighted, int parallel, volatile bool* pInterrupt);  // parallel: 0 or 1 (sequential);  < 0 (cores-2 threads); > 1 (the exact num of threads)
	void GenerateTree(int* index, int nvars, bool isWeighted, volatile bool* pInterrupt);
#endif
	int PredictClass(TrainingSet* training_set,int tuple);
	map<int,int> predictClassMapForOneInstance(TrainingSet* training_set,int tuple);
	vector<double> predictAprobVecForOneInstance(TrainingSet* training_set,int tuple);
	vector<double> predictWaprobVecForOneInstance(TrainingSet* training_set,int tuple);
	double GetErrorRate(TrainingSet* training_set,vector<int> training_set_index);
	vector<vector<int> > GetRandomTrainingSet();
	void GetEachTreeErrorRate(TrainingSet* training_set,vector<int> training_set_index);
	void PrintTrees();
	vector<vector<int> > get_bagging_set();

	int get_trees_num();
	void set_trees_num(int num);
	vector<vector<bool> > get_selected_status_();

	/*
	 * set and get oob estimated information,it is temporary
	 */
	void set_OOB_error_rate_(double rate);
	double get_OOB_error_rate_();
	void set_strength_(double strength);
	double get_strength();
	void set_correlation_(double correlation);
	double get_correlation_();
	void set_c_s2_(double c_s2);
	double get_c_s2_();
	void PredictClassToFile(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper, string file_name);
	vector< map<string, int> > PredictClassMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper);// for R package
	vector< map<string, double> > PredictAprobMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper);
	vector< map<string, double> > PredictWaprobMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper);
	

	Rcpp::List save_model(bool isPart = false);
	void save_one_tree(Node* root, vector<vector<double> >& tree);
	
	vector<int> getOOBset(int i) { return OOBSet[i]; }


};

#endif
