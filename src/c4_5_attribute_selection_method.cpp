/*
 * c4_5_attribute_selection_method.cpp
 *
 *  Created on: 10 Feb, 2012
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */
#include "c4_5_attribute_selection_method.h"

const int C4_5AttributeSelectionMethod::training_set_minimum_ = 2;

C4_5AttributeSelectionMethod::C4_5AttributeSelectionMethod(
		TrainingSet* training_set, vector<int>& training_set_index,
		vector<int>& attribute_list, int seed) :
		AttributeSelectionMethod(training_set, training_set_index,
				attribute_list) {
	this->info1_ = this->CalculateInfo(training_set_index);
	this->seed_num_ = seed;
	this->current_attribute_ = -1;
}


void C4_5AttributeSelectionMethod::set_status_(bool status) {
	this->status_ = status;
}

bool C4_5AttributeSelectionMethod::get_status_() {
	return this->status_;
}

void C4_5AttributeSelectionMethod::set_info_gain_(int attribute,
		double info_gain) {
	this->info_gain_.insert(map<int, double>::value_type(attribute, info_gain));
}

double C4_5AttributeSelectionMethod::get_info_gain_(int attribute) {
	return this->info_gain_[attribute];
}

void C4_5AttributeSelectionMethod::set_split_info_(int attribute,
		double split_info) {
	this->split_info_.insert(
			map<int, double>::value_type(attribute, split_info));
}

double C4_5AttributeSelectionMethod::get_split_info_(int attribute) {
	return this->split_info_[attribute];
}

void C4_5AttributeSelectionMethod::set_splited_training_set(int attribute, map<int, vector<int> > splited_training_set_index) {
	this->splited_training_set_.insert(map<int, map<int, vector<int> > >::value_type(attribute, splited_training_set_index));
}

map<int, vector<int> > C4_5AttributeSelectionMethod::get_splited_training_set_(
		int attribute) {
	return this->splited_training_set_[attribute];
}

void C4_5AttributeSelectionMethod::set_split_value_(int attribute,
		double split_value) {
	this->split_value_.insert(
			map<int, double>::value_type(attribute, split_value));
}

double C4_5AttributeSelectionMethod::get_split_value(int attribute) {
	return this->split_value_[attribute];
}

/*
 * return entropy of the training set with <training_set_index>
 */
double C4_5AttributeSelectionMethod::CalculateInfo(const vector<int>& training_set_index) {
//	double info = 0;
//	double base = 2;
	vector<int> class_distribution = this->get_training_set_()->GetClassesNum(training_set_index);
	return this->CalculateInfoByClassNum(class_distribution,training_set_index.size());
}



/*
 * calculate corresponding information
 * if split by discrete variable <atrribute>
 */
bool C4_5AttributeSelectionMethod::HandleDiscreteAttribute(int attribute) {
	/*
	 * if no more than 2 child nodes contain at least <training_set_minimum_>
	 * instances, don't split training set by this attribute
	 */
	map<int, vector<int> > mapper = this->training_set_->SplitByDiscreteAttribute(this->training_set_index_, attribute);
	int count = 0;
	for (map<int, vector<int> >::iterator iter = mapper.begin(); iter != mapper.end(); ++iter) {
		if (int(iter->second.size()) >= get_training_set_minimum_())
			count++;
	}
	if (count < 2)
		return false;

	/*
	 * calculate corresponding information gain and split entropy
	 * while split by this variable <atrribute>
	 */
	double info2 = 0;
	this->splited_training_set_.insert(map<int, map<int, vector<int> > >::value_type(attribute, mapper));
	double temp_ratio;
	double split_info = 0;
	for (map<int, vector<int> >::iterator iter = mapper.begin();iter != mapper.end(); ++iter) {
		int vector_size = iter->second.size();
		if (vector_size != 0) {
			temp_ratio = (double) vector_size / (double) this->case_num_;
			split_info += (-(temp_ratio) * (log(temp_ratio) / log(2.0)));
			info2 += (temp_ratio) * (this->CalculateInfo(iter->second));
		}
	}

	double info_gain = this->info1_ - info2;
	this->set_info_gain_(attribute, info_gain);
	this->set_split_info_(attribute, split_info);
	return true;
}



bool C4_5AttributeSelectionMethod::HandleContinuousAttribute(int attribute) {

	this->current_attribute_ = attribute;
	vector<int> ordered_training_set_index = this->training_set_index_;
	TrainingSet* training_set = this->get_training_set_();
	if (this->case_num_ < 4)
		return false;

	this->SortTrainingSet(ordered_training_set_index);

	int class_attribute = training_set->GetClassifyAttribute();
	vector<int> mapper_class_num_left;
	for (int j = 0; j < training_set->GetAttributeValueNum(class_attribute); ++j)
		mapper_class_num_left.push_back(0);

	double current_value;
	double next_value;
	double info2;
	bool info2_is_set = false;
	int pos;
	int class_num = training_set->GetAttributeValueNum(class_attribute);

	int min_split = (this->case_num_ * 0.1) / class_num;
	if (min_split > 25) {
		min_split = 25;
	} else if (min_split < get_training_set_minimum_()) {
		min_split = get_training_set_minimum_();
	}

	vector<int> mapper_class_num_right = training_set->GetClassesNum(ordered_training_set_index);
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	for (int i = 0; i < this->case_num_ - min_split + 1; ++i) {
		int class_index = value_matrix[class_attribute][ordered_training_set_index.at(i)].discrete_value_;
		if (i < min_split) {
			;
		} else {
//			next_value = index_value_sort.at(i).value_;
			next_value = value_matrix[attribute][ordered_training_set_index.at(i)].continuous_value_;
			if (current_value != next_value) {
				double new_info2 = ((double) i / (double) this->case_num_) * this->CalculateInfoByClassNum(mapper_class_num_left,i) + ((double) (this->case_num_ - i) / (double) this->case_num_) * this->CalculateInfoByClassNum(mapper_class_num_right,this->case_num_ - i);
				if (info2_is_set) {
					if (new_info2 < info2) {
						info2 = new_info2;
						pos = i - 1;
					}
				} else {
					info2 = new_info2;
					info2_is_set = true;
					pos = i - 1;
				}

			}
		}
		current_value = value_matrix[attribute][ordered_training_set_index.at(i)].continuous_value_;
		mapper_class_num_left.at(class_index) ++;
		mapper_class_num_right.at(class_index) --;
	}

	if(info2_is_set) {
		double info_gain = this->info1_ - info2;
		this->set_info_gain_(attribute, info_gain);
		double split_value = (value_matrix[attribute][ordered_training_set_index.at(pos)].continuous_value_ + value_matrix[attribute][ordered_training_set_index.at(pos + 1)].continuous_value_) / 2;

		this->set_split_value_(attribute, split_value);

		double ratio1 = (double) (pos + 1) / (double) (this->case_num_);
		double ratio2 = (double) (this->case_num_ - pos - 1) / (double) (this->case_num_);
		double split_info = (-ratio1) * (log(ratio1) / log(2.0)) + (-ratio2) * (log(ratio2) / log(2.0));

		this->set_split_info_(attribute, split_info);

		map<int, vector<int> > mapper = training_set->SplitByPositon(ordered_training_set_index, pos);
		this->set_splited_training_set(attribute, mapper);

		return true;

	} else {
		return false;
	}
}

bool C4_5AttributeSelectionMethod::CompStatic(int a, int b) const{
	AttributeValue** value_matrix = training_set_->GetValueMatrixP();
	if(value_matrix[this->current_attribute_][a].continuous_value_ < value_matrix[this->current_attribute_][b].continuous_value_ ){
		return true;
	}else{
		return false;
	}
}

void C4_5AttributeSelectionMethod::SortTrainingSet(vector<int>& training_set_index){
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	sort(training_set_index.begin(), training_set_index.end(), boost::bind(&C4_5AttributeSelectionMethod::CompStatic, this, _1, _2));
#else
	using namespace std::placeholders;
	sort(training_set_index.begin(), training_set_index.end(), bind(&C4_5AttributeSelectionMethod::CompStatic, this, _1, _2));
#endif
#else
	sort(training_set_index.begin(), training_set_index.end(), SelectionComparor(*this));
#endif

}


bool comp(IndexValue a, IndexValue b) {
	if (a.value_ < b.value_) {
		return true;
	} else {
		return false;
	}
}

void C4_5AttributeSelectionMethod::SortTrainingSetByContinuousAttribute(
		vector<IndexValue> &index_value_sort) {
	sort(index_value_sort.begin(), index_value_sort.end(), comp);

}

/*
 * calculate entropy according to number count of each class label
 */
double C4_5AttributeSelectionMethod::CalculateInfoByClassNum(vector<int>& class_num, int all_num) {
	double info = 0;
	vector<int>::iterator iter;
	for (iter = class_num.begin(); iter != class_num.end(); ++iter) {
		if ((*iter) != 0) {
			double ratio = (double) (*iter) / (double) all_num;
			info += (-ratio) * (log(ratio) / log(2.0));
		}
	}
	return info;
}



/*
 * randomly select <subspacesize> variables from <attribute_list>
 * default subspace size is log(n)/log2 + 1 if <subspaceSize> == -1
 */
vector<int> C4_5AttributeSelectionMethod::GetRandomSubSpace(
		vector<int> attribute_list, int subspaceSize) {
	// the result attribute list can't be repeatable
	vector<int> result;
	int attribute_num = attribute_list.size();
	int result_size = subspaceSize;
	if (subspaceSize == -1)
		result_size = log((double) attribute_num) / log(2.0) + 1;

	if (result_size >= attribute_num)
		return attribute_list;

	int j = 0;
	int random_num;
	vector<int>::iterator it = attribute_list.begin();

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	boost::random::mt19937 re(this->seed_num_);
	for (j = 0; j < result_size; ++j) {
		boost::random::uniform_int_distribution<int> uid(0, attribute_num - 1);
#else
	default_random_engine re {this->seed_num_};
	for (j = 0; j < result_size; ++j) {
		uniform_int_distribution<int> uid{0, attribute_num - 1};
#endif
		uid(re);
		random_num = uid(re);
		result.push_back(attribute_list.at(random_num));
		attribute_list.erase(it + random_num);
		it = attribute_list.begin();
		attribute_num = attribute_list.size();
	}
#else
	for (j = 0; j < result_size; ++j) {
		srand(j + this->seed_num_);
		random_num = rand() % attribute_num;
		result.push_back(attribute_list.at(random_num));
		attribute_list.erase(it + random_num);
		it = attribute_list.begin();
		attribute_num = attribute_list.size();
	}
#endif
	return result;

}

// default subspace size is log(n)/log2 + 1
vector<int> C4_5AttributeSelectionMethod::GetRandomSubSpace(
		vector<int> attribute_list) {
	// the result attribute list can't be repeatable
	int result_size = log((double) attribute_list.size()) / log(2.0) + 1;
	return GetRandomSubSpace(
			attribute_list, result_size);

}

void C4_5AttributeSelectionMethod::ExecuteSelectionByIGR(int nvars, volatile bool* pInterrupt) {
	/*
	 * calculate all information gain when split by any one of the variables
	 * from the randomly selected subspace of size <nvars>
	 */
	this->set_status_(false);
	for (vector<int>::iterator iter = this->attribute_list_.begin(); iter != this->attribute_list_.end(); ++iter) {
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
		if (*pInterrupt)
			return;
#else
		if (check_interrupt())
		    throw interrupt_exception("The random forest model building is interrupted.");
#endif

		if (this->training_set_->GetAttributeType(*iter) == DISCRETE) {
			if (this->HandleDiscreteAttribute(*iter))
				this->set_status_(true);
		} else {
			if (this->HandleContinuousAttribute(*iter))
				this->set_status_(true);
		}
	}
	if (!this->get_status_())
		return;

	/*
	 * initialization
	 */
	double total_info_gain = 0;
	for (map<int, double>::iterator iter_info_gain = this->info_gain_.begin();iter_info_gain != this->info_gain_.end(); ++iter_info_gain)
		total_info_gain += iter_info_gain->second;

	double average_info_gain = total_info_gain / (double) (this->info_gain_.size());

	/*
	 * following code is IGR weighting method
	 */
	bool is_set_attribute = false;
	vector<int> alternative_attributes;
	vector<double> alternative_gain_ratio;
	for (map<int, double>::iterator iter_info_gain = this->info_gain_.begin(); iter_info_gain != this->info_gain_.end(); ++iter_info_gain) {
			/*the average_info_gain minus 0.001 to avoid the situation
			 where all the info gain is the same */
		if (iter_info_gain->second >= average_info_gain - 0.001) {
			if (this->get_split_info_(iter_info_gain->first) > 0) {
				double temp_gain_ratio = this->get_info_gain_(iter_info_gain->first) / this->get_split_info_(iter_info_gain->first);
				alternative_attributes.push_back(iter_info_gain->first);
				alternative_gain_ratio.push_back(temp_gain_ratio);
				is_set_attribute = true;
			}
		}
	}

	if(alternative_attributes.size() == 0)
		throw std::range_error("alternative_attributes vector is empty");

	IGR igr(seed_num_+1);
	igr.CalculateWeight(alternative_gain_ratio, nvars, pInterrupt);

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
	if (*pInterrupt)
		return;
#else
	// check interruption
	if (check_interrupt())
	    throw interrupt_exception("The random forest model building is interrupted.");
#endif

	int attribute = alternative_attributes.at(igr.GetSelectedResult());

	if (!is_set_attribute)
		attribute = this->info_gain_.begin()->first;

	this->set_attribute_(attribute);

	if (this->get_training_set_()->GetAttributeType(attribute) == CONTINUOUS)
		this->set_split_value_result_(this->get_split_value(attribute));

	this->set_attribute_info_gain_(this->info_gain_[attribute]);
	this->set_splited_training_set_result_(this->get_splited_training_set_(attribute));
}



/*
 * Problem:when we use c4.5 to construct decision tree, the selected attribute information gain
 * is always greater than the average information gain over all attributes
 * How IGR deal with it ?
 */


void C4_5AttributeSelectionMethod::ExecuteSelection(int nvars, volatile bool* pInterrupt) {
	/*
	 * use the random subspace of all features
	 */

	/*
	 * calculate all information gain when split by any one of the variables
	 * from the randomly selected subspace of size <nvars>
	 */

	vector<int> attribute_list = this->GetRandomSubSpace(this->get_attribute_list_(), nvars);

	/*
	 * use the all features
	 */
	//vector<int> attribute_list = this->get_attribute_list_();

	TrainingSet* training_set = this->get_training_set_();
	vector<int>::iterator iter;
	this->set_status_(false);
	for (iter = attribute_list.begin(); iter != attribute_list.end(); ++iter) {

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
		if (*pInterrupt)
			return;
#else
		// check interruption
		if (check_interrupt())
		    throw interrupt_exception("The random forest model building is interrupted.");
#endif

		if (training_set->GetAttributeType(*iter) == DISCRETE) {
			if (this->HandleDiscreteAttribute(*iter)) {
				this->set_status_(true);
			}
		} else {
			if (this->HandleContinuousAttribute(*iter)) {
				this->set_status_(true);
			}
		}
	}

	if (!this->get_status_()) {
		return;
	}

	/*
	 * find the best variable
	 */

	int mapper_info_gain_size = this->info_gain_.size();
	map<int, double>::iterator iter_info_gain;
	iter_info_gain = this->info_gain_.begin();
	double total_info_gain = 0;
	for (iter_info_gain = this->info_gain_.begin();iter_info_gain != this->info_gain_.end(); ++iter_info_gain) {
		total_info_gain += iter_info_gain->second;
	}
	double average_info_gain = total_info_gain / (double) (mapper_info_gain_size);
	double gain_ratio;
	double temp_gain_ratio;
	bool is_set_gain_ratio = false;
	int attribute;
	bool is_set_attribute = false;
//	double split_info; //just for test
	for (iter_info_gain = this->info_gain_.begin(); iter_info_gain != this->info_gain_.end(); ++iter_info_gain) {
		/*the average_info_gain minus 0.001 to avoid the situation
		 where all the info gain is the same */

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
		if (*pInterrupt)
			return;
#else
		// check interruption
		if (check_interrupt())
		    throw interrupt_exception("The random forest model building is interrupted.");
#endif

		if (iter_info_gain->second >= average_info_gain - 0.001) {
			if (this->get_split_info_(iter_info_gain->first) > 0) {
//				split_info = get_split_info_(iter_info_gain->first); //just for test
				temp_gain_ratio = this->get_info_gain_(iter_info_gain->first)
						/ this->get_split_info_(iter_info_gain->first);
				if (is_set_gain_ratio) {
					if (temp_gain_ratio > gain_ratio) {
						gain_ratio = temp_gain_ratio;
						attribute = iter_info_gain->first;
						is_set_attribute = true;
					}
				} else {
					gain_ratio = temp_gain_ratio;
					attribute = iter_info_gain->first;
					is_set_gain_ratio = true;
					is_set_attribute = true;
				}
			}
		}
	}
	if (!is_set_attribute) {
		attribute = this->info_gain_.begin()->first;
	}
	this->set_attribute_(attribute);
	if (this->get_training_set_()->GetAttributeType(attribute) == CONTINUOUS) {
		this->set_split_value_result_(this->get_split_value(attribute));
	}
	this->set_attribute_info_gain_(this->info_gain_[attribute]);
	this->set_splited_training_set_result_(this->get_splited_training_set_(attribute));
}

