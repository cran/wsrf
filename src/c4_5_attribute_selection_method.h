/*
 * c4_5_attribute_selection_method.h
 *
 *  Created on: 10 Feb, 2012
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#ifndef C4_5_ATTRIBUTE_SELECTION_METHOD_H_
#define C4_5_ATTRIBUTE_SELECTION_METHOD_H_
#include<ctime>
#include<cmath>
#include"attribute_selection_method.h"
#include"utility.h"
#include<algorithm>

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
#include <boost/bind.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#else
#include<random>
#endif
#endif

#include<functional>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include"IGR.h"


class C4_5AttributeSelectionMethod:public AttributeSelectionMethod{
private:
	int current_attribute_;
	int seed_num_;
	bool status_;  // attribute selection progress flag
	const static int training_set_minimum_;    // threshold for minimum child node size
	map<int,double> info_gain_;  // information gain for each variable
	map<int,double> split_info_;  // split entropy for each variable
	/*
	 * child nodes split by each variable
	 * <variable> : "<value> : <instances with that value>"
	 */
	map<int,map<int,vector<int> > >splited_training_set_;
	map<int,double> split_value_;  // <variable> : <optimal split value>
	double info1_;  // entropy of this node

public:

	C4_5AttributeSelectionMethod(TrainingSet*,vector<int>&, vector<int>&, int);
	void set_status_(bool status);
	bool get_status_();
	static int get_training_set_minimum_(){
		return training_set_minimum_;
	}

	void set_info_gain_(int attribute,double info_gain);
	double get_info_gain_(int attribute);
	void set_split_info_(int attribute,double split_info);
	double get_split_info_(int attribute);
	void set_splited_training_set(int attribute,map<int,vector<int> > splited_training_set_index);
	map<int,vector<int> > get_splited_training_set_(int attribute);
	void set_split_value_(int attribute,double split_value);
	double get_split_value(int attribute);



	double CalculateInfo(const vector<int>& training_set_index);

	double CalculateContinuousAttributeInfoGain(int attribute);
	bool HandleContinuousAttribute(int attribute);
	void SortTrainingSetByContinuousAttribute(vector<IndexValue>& index_value_sort);
	bool CompStatic(int a, int b) const;
	void SortTrainingSet(vector<int>& training_set_index);
	double CalculateInfoByClassNum(vector<int>& class_num,int all_num);

	double CalculateDiscreteAttributeInfoGain(int attribute);
	bool HandleDiscreteAttribute(int attribute);

	void ExecuteSelection(int nvars, volatile bool* pInterrupt);  // Breiman's classic method
	void ExecuteSelectionByIGR(int nvars, volatile bool* pInterrupt); // IGR weight method


	vector<int> GetRandomSubSpace(vector<int> attribute_list);
	vector<int> GetRandomSubSpace(vector<int> attribute_list, int subspaceSize);
	void TestGetRandomSubSpace();

	struct SelectionComparor {
		const C4_5AttributeSelectionMethod& m_pt;
		SelectionComparor(const C4_5AttributeSelectionMethod& pt): m_pt(pt) {}
		bool operator()(int a, int b) {
			return m_pt.CompStatic(a, b);
		}
	};


};

#endif

