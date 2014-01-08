/*
 * node.h
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#ifndef NODE_H_
#define NODE_H_
#include"utility.h"
#include<Rcpp.h>

#ifdef WSRF_USE_BOOST
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/foreach.hpp>
#endif

class Node{
private:

	NodeType type_;  // type of this node
	int case_num_;   //added for result file
	int id_;   //added for result file
	int level_; //the level which the node in

	int attribute_;
	AttributeType attribute_type_;
	vector<Node*> child_node_vec_;
	vector<int> child_node_id_;
	/*
	 * following attributes is for internal node
	 */
	double split_value_;

	double info_gain_; //added for result file

	/*
	 * following attributes is for leaf node
	 */
	int class_;  // class label of the leaf node
	double purity_;

	vector<int> classNums;
	vector<double> classDistributions;

public:
	Node(int id);
	//Generate a node object for node information
	Node(vector<double> node_info);
	vector<int> get_child_node_id_();
	
	void set_type_(NodeType);
	NodeType GetNodeType();
	void set_case_num_(int case_num);
	int get_case_num_();
	int get_id_();

	void set_attribute_(int attribute);
	int get_attribute_();
	void set_attribute_type_(AttributeType attribute_type);
	void set_split_value(double split_value);
	double get_split_value();
	void set_child_node_(int, Node*);
	void set_all_child_nodes_(vector<Node*> child_nodes);
	vector<Node*> get_all_child_nodes_();
	Node* get_child_node_(int value);

	void setClassNums(vector<int>);
	vector<int> getClassNums();
	vector<double> getClassDistributions();

	/*
	 * the following function is for result data file
	 */
	void set_info_gain_(double info_gain);
	double get_info_gain();

	void set_class_(int class_);
	int get_class_();
	void set_purity_(double purity);
	double get_purity_();

	/*
	 * store the node infomation in a vector<double>
	 */
	vector<double> get_node_info();

	void deleteChild();
};

#endif
