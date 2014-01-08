/*
 * RFDeployment.h
 *
 *  Created on: 28 Mar, 2012
 *      Author: meng
 */

#ifndef RFDEPLOYMENT_H_
#define RFDEPLOYMENT_H_
#include"utility.h"
#include"random_forests.h"
#include"decision_tree.h"
#include<cmath>
#include<fstream>


using namespace std;

struct confusion_matrix {
	int true_p_;
	int false_n_;
	int false_p_;
	int true_n_;
};

class RFDeployment {
private:
	RandomForests* RF_;
	vector<DecisionTree*> random_forests_;  // = RF->get_trees()
	vector<confusion_matrix> c_matrix_;  // confusion matrix for each label
	vector<vector<int> > OOB_confusion_matrix_;

	vector<int> max_j_;
	double OneExpection;
	vector<vector<int> > OOB_predictor_;
	vector<int> OOB_predictor_result_;
	vector<vector<double> > cor_vec_; // correlation between each two trees
	double OOB_error_rate_;
	double RF_strength_;
	double RF_correlation_;
	vector<double> trees_strength_;





public:
	RFDeployment();
	RFDeployment(RandomForests* RF);
	int PredictClass(TrainingSet* training_set,int tuple);
	double GetErrorRate(TrainingSet* training_set,vector<int> training_set_index);
	void GenerateConfusionMatrix(TrainingSet* training_set,vector<int> training_set_index);
	vector<vector<double> > get_cor_vec_();

	vector<double> GetEachTreeStrength();
	void CalculateOOBPredictor(TrainingSet* training_set);
	void CalculateOOBPredictorResult(TrainingSet* training_set);
	void CalculateTheOOBProportion(TrainingSet* training_set);
	void CalculateOOBErrorRate(TrainingSet* training_set);
	double get_OOB_error_rate_();

	void CalculateRFStrength(TrainingSet* training_set);
	double get_RF_strength_();

	void CalculateRFCorrelation(TrainingSet* training_set);
	double get_RF_correlation();

	void CalculateOOBConfusionMatrix(TrainingSet* training_set);
	vector<vector<int> > get_OOB_confusion_matrix_();
	void CalculateEachTreeOOBStrength(TrainingSet* training_set);

	double CalculateTwoTreesCorrelation(int a, int b, TrainingSet* training_set);
	void CalculateCorBetweenEachTwoTree(TrainingSet* training_set);
	

};

#endif /* RFDEPLOYMENT_H_ */
