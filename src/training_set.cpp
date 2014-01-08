/*
 * training_set.cpp
 *
 *  Created on: 29 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */
#include"training_set.h"

TrainingSet::~TrainingSet() {
	if (value_matrix_ != 0) {
		int attribute_num = this->attribute_value_mapper_->GetAttributeNum();
		for (int i = 0; i < attribute_num; i++)
			if (value_matrix_[i])
				free(value_matrix_[i]);

		free(value_matrix_);
	}

};

void TrainingSet::set_attribute_value_mapper(
		AttributeValueMapper* attribute_value_mapper) {
	this->attribute_value_mapper_ = attribute_value_mapper;
}

int TrainingSet::get_training_set_num_() {
	return this->training_set_num_;
}


int TrainingSet::GetClassifyAttribute() {
	return this->attribute_value_mapper_->GetClassifyAttribute();
}

AttributeType TrainingSet::GetAttributeType(int attribute) {
	return this->attribute_value_mapper_->GetAttributeType(attribute);
}

int TrainingSet::GetAttributeNum() {
	return this->attribute_value_mapper_->GetAttributeNum();
}

/*
 * return variable index list except target variable
 */
vector<int> TrainingSet::GetNormalAttributes() {
	vector<int> normal_attributes;
	int i;
	int attribute_num = this->GetAttributeNum();
	int classify_attribute = this->GetClassifyAttribute();
	for (i = 0; i < attribute_num; ++i) {
		if (i == classify_attribute) {
			continue;
		} else {
			normal_attributes.push_back(i);
		}
	}
	return normal_attributes;
}

int TrainingSet::GetAttributeValueNum(int attribute) {
	return this->attribute_value_mapper_->GetAttributeValueNum(attribute);
}

/*
 * return label ratio in training set with training_set_index
 */
map<int, double> TrainingSet::GetClassDistribution(
		vector<int> training_set_index) {
	int classify_attribute = this->GetClassifyAttribute();
	int training_set_num = training_set_index.size();
	map<int, int> mapper;
	int i;
	int class_index;
	for (i = 0; i < training_set_num; i++) {
		class_index =
				this->value_matrix_[classify_attribute][training_set_index.at(i)].discrete_value_;
		if (mapper.find(class_index) == mapper.end()) {
			mapper.insert(map<int, int>::value_type(class_index, 1));
		} else {
			mapper[class_index]++;
		}
	}
	map<int, double> result;
	for (map<int, int>::iterator iter = mapper.begin(); iter != mapper.end();
			++iter) {
		result.insert(
				map<int, double>::value_type(iter->first,
						(double) iter->second / (double) training_set_num));
	}
	return result;
}

/*
 * return the majority class index within training set with training_set_index
 */
int TrainingSet::GetTheMostClass(const vector<int>& training_set_index) {
	int classify_attribute = this->GetClassifyAttribute();
	int training_set_num = training_set_index.size();
	map<int, int> mapper;
	int i;
	int class_index;
	for (i = 0; i < training_set_num; i++) {
		//class_index=this->training_set_.at(training_set_index.at(i)).GetClass(classify_attribute);
		class_index = this->value_matrix_[classify_attribute][training_set_index.at(i)].discrete_value_;
		if (mapper.find(class_index) == mapper.end()) {
			mapper.insert(map<int, int>::value_type(class_index, 1));
		} else {
			mapper[class_index]++;
		}
	}
	map<int, int>::iterator iter;
	int temp_num = 0;
	int result = 0;
	for (iter = mapper.begin(); iter != mapper.end(); ++iter) {
		if (iter->second > temp_num) {
			temp_num = iter->second;
			result = iter->first;
		}
	}
	return result;
}

/**
 * return whether all instances belong to the same classification
 */
bool TrainingSet::IsTrainingSetSameClass(const vector<int>& training_set_index) {
	int training_set_num = training_set_index.size();
	if (training_set_num == 0) {
		throw std::range_error(	"empty set in function: IsTrainingSetSameClass");
	} else if (training_set_num == 1) {
		return true;
	} else {
		int target_attribute = this->GetClassifyAttribute();
		int i = 0;
		//int class_one=this->training_set_.at(training_set_index.at(i)).GetClass(this->GetClassifyAttribute());
		int class_one = this->value_matrix_[target_attribute][training_set_index.at(i)].discrete_value_;
		i++;
		while (i < training_set_num) {
			if (class_one != this->value_matrix_[target_attribute][training_set_index.at(i)].discrete_value_) {
				return false;
			}
			i++;
		}
		return true;

	}
}



/*
 * return a mapping table which elements are
 * <value>:<index list of instances with that same value> pairs
 * where values are possible value of that variable
 */
map<int, vector<int> > TrainingSet::SplitByDiscreteAttribute(vector<int>& training_set_index, int attribute) {
	vector<int> temp_vec;
	map<int, vector<int> > result;
	int i;
	for (i = 0; i < this->GetAttributeValueNum(attribute); ++i) {
		result.insert(map<int, vector<int> >::value_type(i, temp_vec));
	}
	int training_set_num = training_set_index.size();
	for (i = 0; i < training_set_num; ++i) {
		int attribute_value = this->value_matrix_[attribute][training_set_index.at(i)].discrete_value_;
		result[attribute_value].push_back(training_set_index.at(i));
	}
	return result;

}

/*
 * separate training set with <training_set_index> into two parts
 * the split point is at index <pos>
 */
map<int, vector<int> > TrainingSet::SplitByPositon(vector<int>& training_set_index, int pos) {
	int training_set_num = training_set_index.size();
	map<int, vector<int> > mapper;
	if (pos < 0 || pos >= training_set_num) {
		throw std::range_error("wrong in TrainingSet::SplitByPositon");
	} else {
		vector<int> vec;
		int i;
		for (i = 0; i <= pos; ++i) {
			vec.push_back(training_set_index.at(i));
		}
		mapper.insert(map<int, vector<int> >::value_type(0, vec));
		vec.clear();
		for (i = pos + 1; i < training_set_num; ++i) {
			vec.push_back(training_set_index.at(i));
		}
		mapper.insert(map<int, vector<int> >::value_type(1, vec));
	}
	return mapper;
}

/*
 * remove an <attribute> from <attribute_list>
 */
void TrainingSet::DeleteOneAttribute(vector<int>& attribute_list, int attribute) {
	vector<int>::iterator iter;
	for (iter = attribute_list.begin(); iter != attribute_list.end(); ++iter) {
		if ((*iter) == attribute) {
			attribute_list.erase(iter);
			break;
		}
	}

}

string TrainingSet::GetAttributeName(int attribute) {
	return this->attribute_value_mapper_->GetAttributeName(attribute);
}

string TrainingSet::GetAttributeValueName(int attribute, int index) {
	return this->attribute_value_mapper_->GetAttributeValueName(attribute, index);
}

/*
 * return a list of number count of instances classified into
 * different class label within training set with <training_set_index>
 */
vector<int> TrainingSet::GetClassesNum(const vector<int>& training_set_index) {
	int classify_attribute = this->GetClassifyAttribute();
	int training_set_num = training_set_index.size();
	int class_num = this->GetAttributeValueNum(classify_attribute);
	vector<int> mapper;
	int i;
	for (i = 0; i < class_num; ++i) {
		mapper.push_back(0);
	}
	int class_index;
	for (i = 0; i < training_set_num; i++) {
		class_index = this->value_matrix_[classify_attribute][training_set_index.at(i)].discrete_value_;
		mapper.at(class_index)++;
	}
    return mapper;
}

AttributeValue** TrainingSet::GetValueMatrixP() {
	return this->value_matrix_;

}

/**
 * Convert training set of type Rcpp::DataFrame into internal matrix representation, value_matrix_
 * value_matrix_[i][j] represents the i-th variable of the j-th instance in the training set.
 *
 * It should be the first place encountering missing values.
 */
StatusCode TrainingSet::ProduceTrainingSetMatrixRcpp(const Rcpp::DataFrame& ds){
	int column_num = ds.size();
	if (column_num == 0)
		throw std::range_error("Training Set is empty");
	
	//better method to get DataFrame rows number?
	int training_set_num = Rcpp::CharacterVector(ds[0]).size();
	if(training_set_num == 0)
		throw std::range_error("Training Set is empty,the program will exit");
	
	int attribute_num = this->attribute_value_mapper_->GetAttributeNum();
	this->value_matrix_ = (AttributeValue**) malloc(sizeof(AttributeValue*) * attribute_num);
	
	for (int index = 0; index < attribute_num; ++index)
		this->value_matrix_[index] = (AttributeValue*) malloc(sizeof(AttributeValue) * training_set_num);

	for(int i = 0; i < attribute_num; ++ i){
		vector<string> temp_vec = Rcpp::as<vector<string> >(SEXP(Rcpp::CharacterVector(ds[i])));
		if (this->attribute_value_mapper_->GetAttributeType(i) == CONTINUOUS){
		    for(int j = 0; j < training_set_num; ++ j)
				this->value_matrix_[i][j].continuous_value_ = atof(temp_vec.at(j).c_str());
		}else{
			for(int j = 0; j < training_set_num; ++ j){
				map<string, int>::iterator it;
				if (this->attribute_value_mapper_->GetMapperValue(i,temp_vec.at(j), it))
					this->value_matrix_[i][j].discrete_value_ = it->second;
				else
					throw std::range_error("missing value in data file: line " + (j + 1));
			}
		}
		
	}
	this->training_set_num_ = training_set_num;
	return SUCCESS;
}

/*
 * split <str> into different parts with <delim> as separator
 */
StatusCode TrainingSet::split(const string &str, char delim, vector<string> &strVec) {
	string tempWord;
	stringstream tempSS(str);
	while (getline(tempSS, tempWord, delim)) {
		strVec.push_back(tempWord);
	}
	return SUCCESS;

}

/*
 * return value names of a discrete variable, with each names separated by ","
 */
string TrainingSet::GetAllDAttributeValue(int attribute){
	string result = "";
	int attribute_value_num = this->GetAttributeValueNum(attribute);
	int i = 0;
	while(i < attribute_value_num - 1){
		result += this->GetAttributeValueName(attribute,i);
		result += ",";
		i ++;
	}
	result += this->GetAttributeValueName(attribute,i);
	return result;

}

void TrainingSet::set_source_name_(string source_name){
	this->source_name_ = source_name;
}
string TrainingSet::get_source_name_(){
	return this->source_name_;
}

/*
 * return target variable value names, separated by ","
 */
string TrainingSet::GetAllClassName(){
	int target_attribute = this->GetClassifyAttribute();
	int class_num = this->GetAttributeValueNum(target_attribute);
	int i = 0;
	string result = "";
	while(i < class_num - 1){
		result += this->GetAttributeValueName(target_attribute,i);
		result += ",";
		++ i;
	}
	result += this->GetAttributeValueName(target_attribute,i);
	return result;

}
