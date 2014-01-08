/*
 * decision_tree.h
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */
#ifndef DECISION_TREE_H_
#define DECISION_TREE_H_
#include"utility.h"
#include"node.h"
#include"training_set.h"
#include"attribute_selection_method.h"
#include"c4_5_attribute_selection_method.h"
#include<vector>
#include<iostream>
#include<fstream>

using namespace std;

class DecisionTree{
private:

	int seed_num_;
	Node* root_;
	TrainingSet* training_set_;
	AttributeValue** value_matrix_;
	int target_attribute_;
	int node_num_;

	double OOBErrorRate;

public:

	//Generate a decision tree for vector<Node infomation>
	DecisionTree(Rcpp::List nodes_info);
	DecisionTree(TrainingSet* training_set, int seed);
	~DecisionTree();

	int get_node_num() { return node_num_; };

	void set_root_(Node* root);
	Node* get_root_();
	Node* GenerateDecisionTreeByC4_5(vector<int> training_set_index,vector<int> attribute_list, int nvars, bool isWeighted, volatile bool* pInterrupt);
	void TraverseTree(ofstream& f,TrainingSet* training_set,Node* decision_tree);
	Node* PredictClass(TrainingSet* training_set,int index,Node* root);
	double GetErrorRate(TrainingSet* training_set,const vector<int>& training_set_index);

	void setOOBErrorRate(double errorRate);
	double getOOBErrorRate();
};

#endif
