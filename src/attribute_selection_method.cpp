/*
 * attribute_selection_method.cpp
 *
 *  Created on: 2 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */


#include"attribute_selection_method.h"

// TrainingSet* AttributeSelectionMethod::training_set_ = NULL;

AttributeSelectionMethod::AttributeSelectionMethod(TrainingSet* training_set, vector<int>& training_set_index, vector<int>& attribute_list)
: training_set_index_(training_set_index), attribute_list_(attribute_list)
{
	    this->training_set_=training_set;
	    set_training_set_(training_set);
		this->case_num_ = training_set_index.size();
}

void AttributeSelectionMethod::set_training_set_(TrainingSet* training_set){
	this->training_set_ = training_set;
}


TrainingSet* AttributeSelectionMethod::get_training_set_(){
	return this->training_set_;
}

vector<int> AttributeSelectionMethod::get_training_set_index_(){
	return this->training_set_index_;
}

vector<int> AttributeSelectionMethod::get_attribute_list_(){
	return this->attribute_list_;
}

void AttributeSelectionMethod::set_attribute_(int attribute){
	this->attribute_=attribute;
}

int AttributeSelectionMethod::get_attribute_(){
	return this->attribute_;
}

void AttributeSelectionMethod::set_split_value_result_(double split_value){
	this->split_value_result_=split_value;
}

double AttributeSelectionMethod::get_split_value_result_(){
	return this->split_value_result_;
}

void AttributeSelectionMethod::set_splited_training_set_result_(map<int,vector<int> > splited_training_set_index){
	this->splited_training_set_result_=splited_training_set_index;
}

map<int,vector<int> > AttributeSelectionMethod::get_splited_training_set_result_(){
	return this->splited_training_set_result_;
}

AttributeSelectionResult AttributeSelectionMethod::GetSelectionResult(){
	AttributeSelectionResult result;
	result.attribute_=this->attribute_;
	if(this->training_set_->GetAttributeType(this->attribute_)==CONTINUOUS)
		result.split_value_=this->split_value_result_;
#ifdef WSRF_USE_C11
	result.splited_training_set=move(this->splited_training_set_result_);
#else
	result.splited_training_set=this->splited_training_set_result_;
	this->splited_training_set_result_.clear();
#endif
	result.info_gain_ = this->attribute_info_gain_;
	return result;
}

void AttributeSelectionMethod::set_case_num_(int case_num){
	this->case_num_ = case_num;

}
int AttributeSelectionMethod::get_case_num_(){
	return this->case_num_;
}

void AttributeSelectionMethod::set_attribute_info_gain_(double info_gain){
	this->attribute_info_gain_ = info_gain;
}
double AttributeSelectionMethod::get_attribute_info_gain_(){
	return this->attribute_info_gain_;
}
