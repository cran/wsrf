/*
 * decision_tree.cpp
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#include"decision_tree.h"
#include"utility.h"


DecisionTree::DecisionTree(TrainingSet* training_set, int seed){
	this->training_set_ = training_set;
	this->value_matrix_ = training_set->GetValueMatrixP();
	this->target_attribute_ = training_set->GetClassifyAttribute();
	this->seed_num_ = seed;
	this->node_num_ = 0;
}

/*
 * construct a tree from node info list
 * but no corresponding counter process in class DecisionTree
 * while in RandomForests::save_one_tree()
 */
DecisionTree::DecisionTree(Rcpp::List nodes_info){
	vector<Node*> tree;
	Rcpp::List::iterator it;
	for(it = nodes_info.begin(); it != nodes_info.end(); ++ it){
		Node* one_node = new Node(Rcpp::as<vector<double> >(SEXP(*it)));
		tree.push_back(one_node);
	}
	this->root_ = tree.at(0);
	
	
	vector<int> child_node_id;
	vector<int>::iterator child_it;
	
	vector<Node*> child_nodes;
	
	vector<Node*>::iterator node_it;
	
	for(node_it = tree.begin(); node_it != tree.end(); ++ node_it){
		child_node_id = (*node_it)->get_child_node_id_();
		
		for(child_it = child_node_id.begin(); child_it != child_node_id.end(); ++ child_it){
			child_nodes.push_back(tree.at(*child_it - 1));  // Node ID Start from 1
		}
		
		(*node_it)->set_all_child_nodes_(child_nodes);
		child_node_id.clear();
		child_nodes.clear();
	}
}

DecisionTree::~DecisionTree() {
	if (root_) {
		root_->deleteChild();
		delete root_;
	}
}

void DecisionTree::set_root_(Node* root){
	this->root_=root;
}

Node* DecisionTree::get_root_(){
	return this->root_;
}

/**
 * Main tree building function
 * in a recursive fashion
 *
 */
Node* DecisionTree::GenerateDecisionTreeByC4_5(vector<int> training_set_index,vector<int> attribute_list, int nvars, bool isWeighted, volatile bool* pInterrupt){

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
	if (*pInterrupt) {
		return NULL;
	}
#else
	// check interruption
	if (check_interrupt())
	    throw interrupt_exception("The random forest model building is interrupted.");
#endif

	if(training_set_index.size()==0){

		throw std::range_error("error in decision_tree.cpp function GenerateTreeByC4_5:training set is empty");

	}else if(this->training_set_->IsTrainingSetSameClass(training_set_index)){
		/*
		 * all instances have the same class label
		 */
		Node* node = new Node(++node_num_);
		node->set_type_(LEAFNODE);
        int class_index = this->value_matrix_[this->target_attribute_][training_set_index.at(0)].discrete_value_;
        node->set_class_(class_index);
        node->set_case_num_(training_set_index.size());
        node->setClassNums(this->training_set_->GetClassesNum(training_set_index));
		return node;
	}else if(attribute_list.size()==0){
		int majority_class=this->training_set_->GetTheMostClass(training_set_index);
		Node* node = new Node(++node_num_);
		node->set_type_(LEAFNODE);
		node->set_class_(majority_class);
		node->set_case_num_(training_set_index.size());
		node->setClassNums(this->training_set_->GetClassesNum(training_set_index));
		return node;
	}else {
		C4_5AttributeSelectionMethod* method = new C4_5AttributeSelectionMethod(this->training_set_,training_set_index,attribute_list, seed_num_++);
		if(isWeighted)
			method->ExecuteSelectionByIGR(nvars, pInterrupt);
		else
			method->ExecuteSelection(nvars, pInterrupt);

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
		if (*pInterrupt) {
			delete method;
			return NULL;
		}
#else
		// check interruption
		if (check_interrupt())
		    throw interrupt_exception("The random forest model building is interrupted.");
#endif

		if(!method->get_status_()){
			// if no better attribute selected
			int majority_class=this->training_set_->GetTheMostClass(training_set_index);
			Node* node = new Node(++node_num_);
			node->set_type_(LEAFNODE);
			node->set_class_(majority_class);
			node->set_case_num_(training_set_index.size());
			node->setClassNums(this->training_set_->GetClassesNum(training_set_index));
			delete method;
			return node;
		}else{
			AttributeSelectionResult result = method->GetSelectionResult();
			delete method;
			Node* node = new Node(++node_num_);
			node->set_type_(INTERNALNODE);
			int attribute=result.attribute_;
			node->set_attribute_(attribute);
			node->set_case_num_(training_set_index.size()); // added for result file
			node->set_info_gain_(result.info_gain_);

			if(this->training_set_->GetAttributeType(attribute)==DISCRETE){
				node->set_attribute_type_(DISCRETE);
				this->training_set_->DeleteOneAttribute(attribute_list,attribute);
				for(map<int,vector<int> >::iterator iter=result.splited_training_set.begin();iter!=result.splited_training_set.end();++iter){
					if(iter->second.size()==0){  // maybe this if branch can be deleted
						Node* leaf_node = new Node(++node_num_);
						leaf_node->set_type_(LEAFNODE);
						leaf_node->set_class_(this->training_set_->GetTheMostClass(training_set_index));
						leaf_node->set_case_num_(0);
						// when the discrete value has no corresponding samples in training set, use parent node sample statistics
						leaf_node->setClassNums(this->training_set_->GetClassesNum(training_set_index));
						node->set_child_node_(iter->first,leaf_node);
					}else{
#ifdef WSRF_USE_C11
						node->set_child_node_(iter->first,GenerateDecisionTreeByC4_5(move(iter->second),attribute_list,nvars,isWeighted, pInterrupt));
#else
						node->set_child_node_(iter->first,GenerateDecisionTreeByC4_5(iter->second,attribute_list,nvars,isWeighted, pInterrupt));
						iter->second.clear();
#endif
					}
				}
				return node;
			} else{
				node->set_attribute_type_(CONTINUOUS);
				map<int,vector<int> >::iterator iter;
				for(iter=result.splited_training_set.begin();iter!=result.splited_training_set.end();++iter){
					node->set_split_value(result.split_value_);
					//internal_node->set_case_num_(iter->second.size());
#ifdef WSRF_USE_C11
					node->set_child_node_(iter->first,GenerateDecisionTreeByC4_5(move(iter->second),attribute_list,nvars,isWeighted, pInterrupt));
#else
					node->set_child_node_(iter->first,GenerateDecisionTreeByC4_5(iter->second,attribute_list,nvars,isWeighted, pInterrupt));
					iter->second.clear();
#endif
				}
				return node;
			
			}
		}
	}
}


/*
 * return leaf node to which the instance belongs.
 * index : is the index of the instance in the training set
 */
Node* DecisionTree::PredictClass(TrainingSet* training_set,int index, Node* node){
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	if(node->GetNodeType()==LEAFNODE){
		return node;
	}else{
		int attribute=node->get_attribute_();
		if(training_set->GetAttributeType(attribute)==DISCRETE){                  
			return PredictClass(training_set,index,node->get_child_node_(value_matrix[attribute][index].discrete_value_));
		}else{
			//double value=training_set->At(index).At(attribute).get_continuous_value();
			double value = value_matrix[attribute][index].continuous_value_;
			if(value <= node->get_split_value()){
				return PredictClass(training_set,index,node->get_child_node_(0));
			}else{
				return PredictClass(training_set,index,node->get_child_node_(1));
			}
		}
	}
}

/*
 * return error rate for classifying training set
 */
double DecisionTree::GetErrorRate(TrainingSet* training_set,const vector<int>& training_set_index){
	vector<int>::const_iterator iter;
	int error=0;
	Node* leaf_node;
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int target_attribute = training_set->GetClassifyAttribute();
	for(iter=training_set_index.begin();iter!=training_set_index.end();++iter){
		leaf_node=this->PredictClass(training_set,*iter,this->get_root_());
		if(leaf_node->get_class_() == value_matrix[target_attribute][*iter].discrete_value_){
		}
		else{
			error++;
		}
		
	}
	double error_rate=((double)error)/((double)(training_set_index.size()));
	return error_rate;
}

void DecisionTree::setOOBErrorRate(double errorRate) {
	this->OOBErrorRate = errorRate;
}

double DecisionTree::getOOBErrorRate() {
	return this->OOBErrorRate;
}

