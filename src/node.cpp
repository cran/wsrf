/*
 * node.cpp
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#include"node.h"


Node::Node(int id){
	this->id_ = id;
}

/*
 * see "vector<double> Node::get_node_info()" for details
 */
Node::Node(vector<double> node_info){
	this->id_ = node_info.at(0);
	this->type_ = (NodeType)node_info.at(1);
	if(this->type_  == LEAFNODE){
		this->class_ = node_info.at(2);
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
		for_each(node_info.begin() + 3, node_info.end(), boost::bind(&std::vector<int>::push_back, &this->classNums, boost::lambda::_1));
#else
		for_each(node_info.begin() + 3, node_info.end(), [&](double num) { this->classNums.push_back(num); });
#endif
#else
		for (int i = 3; i < node_info.size(); i++) {
			this->classNums.push_back(*(node_info.begin() + i));
		}
#endif
	}else{
		this->attribute_ = node_info.at(2);
		this->attribute_type_ = (AttributeType)node_info.at(3);
		if(this->attribute_type_ == DISCRETE){
			int i = 4;
			while(i < int(node_info.size())){
				this->child_node_id_.push_back(node_info.at(i));
				++ i;
			}
		}else{
			this->split_value_ = node_info.at(4);
			this->child_node_id_.push_back(node_info.at(5));
			this->child_node_id_.push_back(node_info.at(6));
		}
		
	}
}

void Node::setClassNums(vector<int> classNums) {
	this->classNums = classNums;
}

vector<int> Node::getClassNums() {
	return this->classNums;
}

vector<double> Node::getClassDistributions() {

	if (type_ != LEAFNODE)
		throw range_error("Internal node has no class distributions");

	if (classDistributions.size() <= 0) {
		double sum = 0;
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
		BOOST_FOREACH (int& num, classNums) sum += num;
		BOOST_FOREACH (int& num, classNums) classDistributions.push_back(num/sum);
#else
		for (int& num : classNums) sum += num;
		for (int& num : classNums) classDistributions.push_back(num/sum);
#endif
#else
		for (int i = 0; i < classNums.size(); i++) sum += classNums[i];
		for (int i = 0; i < classNums.size(); i++) classDistributions.push_back(classNums[i]/sum);
#endif
	}

	return classDistributions;
}

void Node::deleteChild() {
	if (child_node_vec_.size() > 0) {
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
		BOOST_FOREACH (Node * child, child_node_vec_) {
#else
		for (auto child : child_node_vec_) {
#endif
			if (child) {
				child->deleteChild();
				delete child;
			}
		}
#else
		for (int i = 0; i < child_node_vec_.size(); i++) {
			if (child_node_vec_[i]) {
				child_node_vec_[i]->deleteChild();
				delete child_node_vec_[i];
			}
		}
#endif
		child_node_vec_.clear();
	}
}

vector<int> Node::get_child_node_id_(){
	return this->child_node_id_;
}

void Node::set_type_(NodeType type){
	this->type_=type;
}

NodeType Node::GetNodeType(){
	return this->type_;
}

void Node::set_case_num_(int case_num){
	this->case_num_ = case_num;
}
int Node::get_case_num_(){
	return this->case_num_;
}
int Node::get_id_(){
	return this->id_;
}

void Node::set_attribute_(int attribute){
	this->attribute_=attribute;
}
int Node::get_attribute_(){
	return this->attribute_;
}

void Node::set_attribute_type_(AttributeType attribute_type){
	this->attribute_type_ = attribute_type;
}

void Node::set_split_value(double split_value){
	this->split_value_=split_value;
}
double Node::get_split_value(){
	return this->split_value_;
}
/*
 * insert a <node> at <attribute>
 */
void  Node::set_child_node_(int attribute,Node* node){
	if (node == 0) return;
	int inc = attribute + 1 - this->child_node_vec_.size();
	while(inc > 0){
		this->child_node_vec_.push_back(NULL);
		inc --;
	}
	this->child_node_vec_.at(attribute) = node;
}
void Node::set_all_child_nodes_(vector<Node*> child_nodes){
	this->child_node_vec_ = child_nodes;
}
Node* Node::get_child_node_(int value){
	return this->child_node_vec_.at(value);
}


vector<Node*> Node::get_all_child_nodes_(){
	return this->child_node_vec_;
}


void Node::set_info_gain_(double info_gain){
	this->info_gain_ = info_gain;
}
double Node::get_info_gain(){
	return this->info_gain_;
}

void Node::set_class_(int class_){
	this->class_=class_;
}
int Node::get_class_(){
	return this->class_;
}

void Node::set_purity_(double purity){
	this->purity_ = purity;
}

double Node::get_purity_(){
	return this->purity_;
}

/*
 * serialize a this node
 *
 * leaf node structure:
 *   1.ID
 *   2.type
 *
 *   3.class label
 *   4.class nums
 *
 * internal node structure:
 *   1.Id
 *   2.type
 *
 *   3.attribute
 *   4.attribute_type
 *   5.split_value (optional,depends on 4)
 *   6.child id
 */
vector<double> Node::get_node_info(){
	/*
	 * leaf node structure:1.ID, 2.type, 3.class label, 4.class nums
	 *
	 * internal node structure:1.Id, 2.type, 3.attribute, 4.attribute_type 5.(optional,depend on 4) split_value
	 *                         6...child id
	 */
	vector<double> node_info;
	node_info.push_back(this->id_);
    node_info.push_back((double) this->type_);
	
	if(this->type_ == LEAFNODE){
		node_info.push_back(this->class_);
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
		BOOST_FOREACH (int num, this->classNums) node_info.push_back(num);
#else
		for (int num : this->classNums) node_info.push_back(num);
#endif
#else
		for (int i = 0; i < classNums.size(); i++) node_info.push_back(classNums[i]);
#endif
		return node_info;
	}

	node_info.push_back(this->attribute_);
	node_info.push_back((double) this->attribute_type_);
	if (this->attribute_type_ == CONTINUOUS) {
		node_info.push_back(this->split_value_);
	}

	vector<Node*>::iterator it;
	for(it = this->child_node_vec_.begin(); it != this->child_node_vec_.end(); ++ it){
		node_info.push_back((*it)->id_);

	}
	return node_info;

}
