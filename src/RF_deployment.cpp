/*
 * RFDeployment.cpp
 *
 *  Created on: 28 Mar, 2012
 *      Author: meng email:qinghan.meng@gmail.com
 *  problem:when a training example participated any tree's construction
 */

#include "RF_deployment.h"
/*
 * #include"MDScaling.h" using MDS(multidimensional scaling) to get trees relative position
 * it is for VisForests
 */
//#include"MDScaling.h"

RFDeployment::RFDeployment() {

}

RFDeployment::RFDeployment(RandomForests* RF) {
	this->RF_ = RF;
	this->random_forests_ = RF->get_trees();
}

/*
 * trees int the forest vote for the class
 */

int RFDeployment::PredictClass(TrainingSet* training_set, int tuple) {
	map<int, int> mapper;
	vector<DecisionTree*>::iterator iter;
	for (iter = this->random_forests_.begin();
			iter != this->random_forests_.end(); ++iter) {
		int target_class = ((*iter)->PredictClass(training_set, tuple, (*iter)->get_root_()))->get_class_();
		if (mapper.find(target_class) == mapper.end()) {
			mapper.insert(map<int, int>::value_type(target_class, 1));
		} else {
			mapper[target_class]++;
		}

	}
	int max = 0;
	int result = 0;
	map<int, int>::iterator map_iter;
	for (map_iter = mapper.begin(); map_iter != mapper.end(); ++map_iter) {
		if (map_iter->second >= max) {
			result = map_iter->first;
			max = map_iter->second;
		}
	}
	return result;
}

/*
 * using a training set to get the random forest error rate
 */
double RFDeployment::GetErrorRate(TrainingSet* training_set, vector<int> training_set_index) {
	int error = 0;
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int target_attribute = training_set->GetClassifyAttribute();
	vector<int>::iterator iter;
	for (iter = training_set_index.begin(); iter != training_set_index.end();
			++iter) {
		if (this->PredictClass(training_set, *iter)
				!= value_matrix[target_attribute][*iter].discrete_value_) {
			error++;
		}
	}
	return ((double) error) / ((double) training_set_index.size());
}

/*
 * following code CalculateOOBPredictorResultis added from 7/20/2012
 */

/*
 * fill <OOB_predictor_>, which is a <training set size> * <number of trees>
 * matrix
 * those instances that are not in one of the tree training set bags will have
 * a predict value at [instance index][tree index] in the matrix <OOB_predictor_>,
 * where -1 indicates <instance index> is in the bag of <tree index>
 */
void RFDeployment::CalculateOOBPredictor(TrainingSet* training_set){
	int training_set_num = training_set->get_training_set_num_();
//	int target_attribute = training_set->GetClassifyAttribute();
	int trees_num = this->RF_->get_trees_num();
	vector<vector<bool> > selected_status = this->RF_->get_selected_status_();

	int i;
	for (i = 0; i < training_set_num; ++i) {
		vector<int> temp_vec(trees_num,-1);
		int j;
		for (j = 0; j < trees_num; ++j) {
			if (selected_status.at(j).at(i) == false) {
			    int predicted_class =(this->random_forests_.at(j)->PredictClass(training_set, i, this->random_forests_.at(j)->get_root_()))->get_class_();
				temp_vec.at(j) = predicted_class;
			}

		}
		this->OOB_predictor_.push_back(temp_vec);
	}
}

/*
 * you should run the CalculateOOBPredictor() function first
 */

/*
* fill <OOB_predictor_result_>, which is a vector contains predict results for
* all instances which are not in at least one of the tree bags. As for those
* which are included in all tree bags, the corresponding value are -1
*/
void RFDeployment::CalculateOOBPredictorResult(TrainingSet* training_set){
	int target_attribute = training_set->GetClassifyAttribute();
	int class_num = training_set->GetAttributeValueNum(target_attribute);
	vector<vector<int> >::iterator it1;
	for(it1  = this->OOB_predictor_.begin(); it1 != this->OOB_predictor_.end(); ++ it1){
		vector<int> temp_vec(class_num,0);
		vector<int>::iterator it2;
		for(it2 = it1->begin(); it2 != it1->end(); ++ it2){
			if(*it2 != -1){
				temp_vec.at(*it2) ++;
			}
		}
		int max = 0;
		int result = -1;
		int i;
		for(i = 0; i < class_num; ++ i){
			if(temp_vec.at(i) > max){
				max = temp_vec.at(i);
				result = i;
			}
		}
		this->OOB_predictor_result_.push_back(result);
	}
}

/*
 * the denominator is the oob training set, not all the training set
 * before you use this function result,you should run CalculateOOBPredictor() first;
 *
 * fill <trees_strength_>, which is the strength of each tree
 * calculated from Out-Of-Bag prediction
 */

void RFDeployment::CalculateEachTreeOOBStrength(TrainingSet* training_set){
	int trees_num = this->RF_->get_trees_num();
	int training_set_num = training_set->get_training_set_num_();
	int target_attribute = training_set->GetClassifyAttribute();
	AttributeValue** value_matrix = training_set->GetValueMatrixP();

	int i;
	for(i = 0; i < trees_num; i ++){
		int numerator = 0;
	    int denominator = 0;
		int j;
		for(j = 0;j < training_set_num; ++ j){
			int predicted_result = this->OOB_predictor_.at(j).at(i);
			if(predicted_result != -1){
				denominator ++;
				if(predicted_result != value_matrix[target_attribute][j].discrete_value_){
					numerator --;
				}else{
					numerator ++;
				}
			}
		}
		this->trees_strength_.push_back((double)numerator / (double)denominator);
	}
}

vector<double> RFDeployment::GetEachTreeStrength(){
	return this->trees_strength_;
}

/*
 * you should run CalculateOOBPredictor() first to fill this->OOB_predictor
 *
 * 在运行此函数之前，必须先运行CalculateOOBPredictor()函数获得this->OOB_predictor
 * 首先得算出两棵树组成的强度
 */
double RFDeployment::CalculateTwoTreesCorrelation(int a, int b, TrainingSet* training_set){
	int training_set_num = training_set->get_training_set_num_();
	int target_attribute = training_set->GetClassifyAttribute();
	int class_num = training_set->GetAttributeValueNum(target_attribute);
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int i;
	vector<vector<int> > tree_class(2, vector<int>(class_num, 0));
	vector<int> max_j(training_set_num, -1);
	vector<double> p_vec;
	vector<int> OOB_num(2, 0);
	vector<int> OOB_correct(2, 0);

	/*
	 * collect out of bag predict information
	 */
	for (i = 0; i < training_set_num; i++) {
		int OOB_classifier = 0;
		int actual_class = value_matrix[target_attribute][i].discrete_value_;
		vector<int> vec(class_num, 0);
		if (this->OOB_predictor_.at(i).at(a) != -1) { // this->OOB_predictor_.at(i).at(a)
			tree_class.at(0).at(this->OOB_predictor_.at(i).at(a)) ++;vec
			.at(this->OOB_predictor_.at(i).at(a)) ++;OOB_num
			.at(0) ++;if
(			actual_class == this->OOB_predictor_.at(i).at(a)) {
				OOB_correct.at(0) ++;
			}
			OOB_classifier++;
		}
		if (this->OOB_predictor_.at(i).at(b) != -1) {
			tree_class.at(1).at(this->OOB_predictor_.at(i).at(b)) ++;vec
			.at(this->OOB_predictor_.at(i).at(b)) ++;OOB_num
			.at(1) ++;if
(			actual_class == this->OOB_predictor_.at(i).at(b)) {
				OOB_correct.at(1) ++;
			}
			OOB_classifier++;
		}
		if (OOB_classifier != 0) {
			int max_error = -1;
			bool is_j_set = false;
			int j = 0;
			int k = 0;
			while (k < class_num) {
				if (k != actual_class) {
					if (!is_j_set) {
						j = k;
						max_error = vec.at(k);
						is_j_set = true;
					} else {
						if (max_error < vec.at(k)) {
							max_error = vec.at(k);
							j = k;
						}
					}
				}
				k++;
			}
			max_j.at(i) = j;
			p_vec.push_back(
					(double) (vec.at(actual_class) - max_error)
							/ (double) OOB_classifier);
		}
	}

	/*
	 * calculate the strength
	 */
	double p_sum = 0.0;
	double e_sum = 0.0;
	vector<double>::iterator it;
	for (it = p_vec.begin(); it != p_vec.end(); ++it) {
		p_sum += *it;
		e_sum += (*it) * (*it);
	}
	double strength = p_sum / (double) (p_vec.size());

	double var = e_sum / (double) (p_vec.size()) - strength * strength;

	int tree_index;
	vector<double> sd(2, 0.0);
	for (tree_index = 0; tree_index < 2; tree_index++) {
		double p1 = (double) OOB_correct.at(tree_index)
				/ (double) OOB_num.at(tree_index);
		int count = 0;
		int t_index;
		if (tree_index == 0) {
			for (t_index = 0; t_index < training_set_num; t_index++) {
				if (max_j.at(t_index) != -1) {
					if (this->OOB_predictor_.at(t_index).at(a)
							== max_j.at(t_index)) {
						count++;
					}
				}

			}
		} else {
			for (t_index = 0; t_index < training_set_num; t_index++) {
				if (max_j.at(t_index) != -1) {
					if (this->OOB_predictor_.at(t_index).at(b)
							== max_j.at(t_index)) {
						count++;
					}
				}
			}
		}
		double p2 = (double) count / (double) OOB_num.at(tree_index);
		double base = p1 + p2 + (p1 - p2) * (p1 - p2);
		double sd_temp = pow(base, 0.5);
		sd.at(tree_index) = sd_temp;
	}
	double e_sd = (sd.at(0) + sd.at(1)) / 2;
	double correlation = var / (e_sd * e_sd);
	return correlation;
}


void RFDeployment::CalculateCorBetweenEachTwoTree(TrainingSet* training_set){
	int trees_num = this->RF_->get_trees_num();
	vector<vector<double> > cor_vec(trees_num,vector<double>(trees_num,0.0));
	int a, b;
	for(a = 0; a < trees_num; a ++){
		for(b = 0; b < trees_num; b ++){
			if(a != b){
				   cor_vec.at(a).at(b) = this->CalculateTwoTreesCorrelation(a,b,training_set);
			}
		}
	}
	this->cor_vec_ = cor_vec;
}

vector<vector<double> > RFDeployment::get_cor_vec_(){
	return this->cor_vec_;
}

/*
 * a two dimension array store the oob confusion matrix,using vector<vector<int> > vec
 * vec.at(i).at(j),i,j represents the class,i is the actual class,j is the predcited class.
 * should run CalculateOOBPredictor() and CalculateOOBPredictorResult() first
 */

void RFDeployment::CalculateOOBConfusionMatrix(TrainingSet* training_set){
	int target_attribute = training_set->GetClassifyAttribute();
	int class_num = training_set->GetAttributeValueNum(target_attribute);
	int training_set_num = training_set->get_training_set_num_();
	this->OOB_confusion_matrix_ = vector<vector<int> >(class_num, vector<int>(class_num,0));
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int i;
	for(i = 0; i < training_set_num; ++ i){
		int actual_class = value_matrix[target_attribute][i].discrete_value_;
		int predicted_class = this->OOB_predictor_result_.at(i);
		/*
		 * maybe an instance does not have a OOB classifier
		 */
		if(predicted_class != -1){
			if (predicted_class == actual_class) {
				this->OOB_confusion_matrix_.at(actual_class).at(
						actual_class) ++;}
            else{
				this->OOB_confusion_matrix_.at(actual_class).at(predicted_class) ++;
			}
		}

	}
}


vector<vector<int> > RFDeployment::get_OOB_confusion_matrix_(){
	return this->OOB_confusion_matrix_;
}

/*
 * we can get the OOB error rate from the OOB confusion matrix.
 * so you should get the OOB confusion matrix first,just run GalculateConfusionMatrix()
 */

void RFDeployment::CalculateOOBErrorRate(TrainingSet* training_set){
	int target_attribute = training_set->GetClassifyAttribute();
	int class_num = training_set->GetAttributeValueNum(target_attribute);
	int OOB_num = 0;
	int correct_num = 0;
	int i;
	for(i = 0; i < class_num; ++ i){
		correct_num += this->OOB_confusion_matrix_.at(i).at(i);
		int j;
		for(j = 0; j < class_num; ++ j){
			OOB_num += this->OOB_confusion_matrix_.at(i).at(j);
		}
	}
	this->OOB_error_rate_ = (double)(OOB_num - correct_num) / (double)OOB_num;
}

double RFDeployment::get_OOB_error_rate_(){
	return this->OOB_error_rate_;
}



void RFDeployment::CalculateRFStrength(TrainingSet* training_set){
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	double sum = 0.0;
	double sum_for_expection = 0.0;
	int training_set_num = training_set->get_training_set_num_();
	int target_attribute = training_set->GetClassifyAttribute();
	int class_num = training_set->GetAttributeValueNum(target_attribute);
	int trees_num = this->RF_->get_trees_num();
	int i;
	int valid_num = 0;
	for(i = 0; i < training_set_num; ++ i){
		/*
		 * find the max_j
		 */
		vector<int> classes_num(class_num,0);
		int OOB_count = 0;
		int j;
		for(j = 0; j < trees_num; ++ j){
			if(this->OOB_predictor_.at(i).at(j) != -1){
				OOB_count ++;
				classes_num.at(this->OOB_predictor_.at(i).at(j)) ++;
			}
		}
        if(OOB_count != 0) {
        	valid_num ++;
			int actual_class = value_matrix[target_attribute][i].discrete_value_;
			int max_j;
			int max;
			bool is_max_j_set = false;
			int k;
			for(k = 0; k < class_num; ++ k) {
				if(k != actual_class) {
					if(!is_max_j_set) {
						max_j = k;
						max = classes_num.at(k);
						is_max_j_set = true;
					} else {
						if(classes_num.at(k) > max){
							max = classes_num.at(k);
							max_j = k;
						}
					}
				}

			}
			this->max_j_.push_back(max_j);
			double temp = (double)(classes_num.at(actual_class) - classes_num.at(max_j)) / (double)(OOB_count);
			sum += temp;
			sum_for_expection += temp * temp;
		}else{
			this->max_j_.push_back(-1);
		}
	}

	this->OneExpection = sum_for_expection / valid_num;
	this->RF_strength_ = sum / valid_num;


}

double RFDeployment::get_RF_strength_(){
	return this->RF_strength_;
}

void RFDeployment::CalculateRFCorrelation(TrainingSet* training_set){
	int training_set_num = training_set->get_training_set_num_();
	int target_attribute = training_set->GetClassifyAttribute();
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int trees_num = this->RF_->get_trees_num();
	vector<vector<bool> > selected_status = this->RF_->get_selected_status_();
	double sd_sum = 0.0;
	int i;
	for(i = 0; i < trees_num; ++ i){
		double p1 = 0.0, p2 = 0.0;
		int i1 = 0, i2 = 0;
		int OOB_instance = 0;
		int j;
		for(j = 0; j < training_set_num; ++ j){
			if(selected_status.at(i).at(j) == false){
				OOB_instance ++;
				int actual_class = value_matrix[target_attribute][j].discrete_value_;
				int max_j = this->max_j_.at(j);
				int predict_class = (this->random_forests_.at(i)->PredictClass(training_set,j,this->random_forests_.at(i)->get_root_()))->get_class_();
				if(predict_class == actual_class){
					i1 ++;
				}else if(predict_class == max_j){
					i2 ++;
				}
			}
		}

		p1 = (double)i1 / (double)OOB_instance;
		p2 = (double)i2 / (double)OOB_instance;
		/*
		 *
		 */
		double base = p1 + p2 + (p1 - p2)*(p1 - p2);
		sd_sum += pow(base,0.5);
	}
	double e_sd = sd_sum / trees_num;
	this->RF_correlation_ = (this->OneExpection - this->RF_strength_*this->RF_strength_) / (e_sd * e_sd);

}

double RFDeployment::get_RF_correlation(){
	return this->RF_correlation_;
}




