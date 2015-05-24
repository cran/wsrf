#ifndef IGR_H_
#define IGR_H_

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iterator>

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#else
#include<random>
#endif
#endif

#include "utility.h"

using namespace std;

class IGR {
private:
    int nvars_;  //subspace size
    unsigned seed_;

    vector<double> weights_;
    vector<int>    wst_;

    const vector<double>& gain_ratio_vec_;


    int weightedSampling(int random_integer);
    vector<int> getRandomWeightedVars();
public:
    IGR(const vector<double>& gain_ratio, int nvars, unsigned seed);

    void normalizeWeight(volatile bool* pInterrupt);
    int  getSelectedIdx();
};

#endif /* IGR_H_ */
