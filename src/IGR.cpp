/*
 * IGR.cpp
 *
 *  Created on: 25 Jul, 2012
 *      Author: meng
 */

#include "IGR.h"
#include<iostream>
IGR::IGR(int seed){
	this->seed = seed;
}

int IGR::FindInternal(int random_integer, vector<int>& weights){
	int size = weights.size();
	int left = 0;
	int right = 0;
	int i;
	for(i = 0; i < size; ++ i){
		left = right;
		right += weights.at(i);
		if(random_integer >= left && random_integer <= right){
			return i;
		}
	}
	//may be the largest right is smaller RAND_MAX,because double to int may lose information
	return i -1;
}

/*
 * generate an integer list of size <size> according to probability
 * that is, select <size> variables by their weights
 */
vector<int> IGR::GenerateRandomIntegerByProbability(int size, vector<double>& probability){

	vector<int> result;
	if (size >= int(probability.size())) {
		for (int i = 0; i < int(probability.size()); i++)
			result.push_back(i);
		return result;
	}

	vector<int> toInt;
	vector<double>::iterator it;
	for(it = probability.begin(); it != probability.end(); ++ it){
		toInt.push_back((int)((*it)*RAND_MAX));
	}

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	boost::random::uniform_int_distribution<int> uid(0, RAND_MAX);
	boost::random::mt19937 re(seed);
#else
	uniform_int_distribution<int> uid{0, RAND_MAX};
	default_random_engine re {seed};
#endif
	for(int i = 0; i < size; ++ i)
		result.push_back(FindInternal(uid(re), toInt));
#else

	int i;
	for(i = 0; i < size; ++ i){
		srand(seed + i);
		result.push_back(FindInternal(rand(), toInt));
	}
#endif
	return result;
}

/*
 * calculate weights of all variables according to their gain ratios
 * the results are in this->weights_
 */
void IGR::CalculateWeight(vector<double> gain_ratio, int subspaceSize, volatile bool* pInterrupt) {
	if (subspaceSize == -1)
		subspaceSize = log((double) gain_ratio.size()) / log(2.0) + 1;
	this->m_ = subspaceSize >= int(gain_ratio.size()) ? int(gain_ratio.size()) : subspaceSize;
	double sum = 0.0;
	vector<double> vec;
	double temp;
	vector<double>::iterator it;
	for (it = gain_ratio.begin(); it != gain_ratio.end(); ++it) {
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
		if (*pInterrupt)
			return;
#else
		// check interruption
		if (check_interrupt())
		    throw interrupt_exception("The random forest model building is interrupted.");
#endif
		temp = pow(*it, 0.5);
		vec.push_back(temp);
		sum += temp;
	}
	if (sum != 0.0) {
		for (it = vec.begin(); it != vec.end(); ++it) {
			this->weights_.push_back(*it / sum);
		}
	} else {
		int i;
		int size = gain_ratio.size();
		for (i = 0; i < size; i++) {
			this->weights_.push_back(1.0 / (double) size);
		}
	}
}

/*
 * select the most weighted variable from < this->m_ > variables that
 * are randomly picked from all varialbes according to their weights
 */
int IGR::GetSelectedResult(){
	vector<int> alternative_result = GenerateRandomIntegerByProbability(this->m_, this->weights_);
	int max;
	bool is_max_set = false;
	int rand_num;
	int i;
	for(i = 0; i < this->m_; i ++){
		rand_num = alternative_result.at(i);
		if(is_max_set){
			if(this->weights_[rand_num] >= this->weights_[max]){
				max = rand_num;
			}
		}else{
			max = rand_num;
			is_max_set = true;
		}
	}
	if(is_max_set){
		return max;
	}else{
		return -1;
	}
}
