/*
 * IGR.h
 *
 *  Created on: 25 Jul, 2012
 *      Author: meng
 */

#ifndef IGR_H_
#define IGR_H_
#include<ctime>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<cmath>
#include"utility.h"

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#else
#include<random>
#endif
#endif
using namespace std;


class IGR {
private:
	int m_; //subspace size
	vector<double> weights_;

	int seed;

	int FindInternal(int random_integer, vector<int>& weights);
	vector<int> GenerateRandomIntegerByProbability(int size, vector<double>& probability);
public:
	IGR(int seed);
	void CalculateWeight(vector<double> gain_ratio, int subspaceSize, volatile bool* pInterrupt);
	int GetSelectedResult();
};

#endif /* IGR_H_ */
