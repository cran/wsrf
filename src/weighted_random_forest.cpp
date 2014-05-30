#include<iostream>
#include<ctime>
#include"utility.h"
#include"attribute_value_mapper.h"
#include"c4_5_attribute_selection_method.h"
#include"training_set.h"
#include"decision_tree.h"
#include"random_forests.h"
#include"RF_deployment.h"
#include"IGR.h"

#include<fstream>
#include"weighted_random_forest.h"

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/exception_ptr.hpp>
#else
#include<thread>
#include<chrono>
#include<future>
#endif
#endif

#include<exception>

#include"decision_tree.h"

/*
 * #include<unistd.h>
 *
 * include the getopt function, you can find the getopt function information in
 * http://www.gnu.org/savannah-checkouts/gnu/libc/manual/html_node/Using-Getopt.html#Using-Getopt
 * -n 80 -f /home/meng/workspace/DataSet/fbis -o -t
 * out of bag 评估应该在生成模型都应该得到评估，不给用户选择。
 * 模型保存应该在所有随机森林数据都产生再保存，注意顺序
 */

#include<unistd.h>
using namespace std;

SEXP WeightedRandomForest(SEXP dsSEXP, SEXP nmSEXP, SEXP nTrees, SEXP nvars, 
			  SEXP isWeighted, SEXP parallel, SEXP seeds, SEXP isPart)
{
  BEGIN_RCPP

  AttributeValueMapper* attribute_value_mapper = new AttributeValueMapper();
  attribute_value_mapper->MapAttributeValueRcpp(Rcpp::DataFrame(nmSEXP));

	TrainingSet* training_set = new TrainingSet();
	training_set->set_attribute_value_mapper(attribute_value_mapper);
	if(training_set->ProduceTrainingSetMatrixRcpp(Rcpp::DataFrame(dsSEXP)) == FAIL)
		throw std::range_error("can't produce the training set !");

	Rcpp::NumericVector n(nTrees);
	Rcpp::IntegerVector sds(seeds);
	RandomForests* random_forests = new RandomForests(training_set, (int)(n[0]), sds);
	Rcpp::NumericVector m(nvars);
	Rcpp::NumericVector iW(isWeighted);
	Rcpp::NumericVector par(parallel);
	Rcpp::NumericVector iPt(isPart);

#if defined WSRF_USE_C11 || defined WSRF_USE_BOOST
	/**
	 * create a thread for model building
	 * leave main thread for interrupt check once per 1 seconds
	 */

	/*
	 * <interrupt> is used to inform each thread of no need to continue, but has 2 roles:
	 *  1. represents user interrupt
	 *  2. represents a exception has been thrown from one tree builder
	 */
	volatile bool interrupt = false;
#ifdef WSRF_USE_BOOST
	boost::packaged_task<void> pt(boost::bind(&RandomForests::GenerateRF, random_forests, (int(m[0])),(bool(iW[0])), (int(par[0])), &interrupt));
	boost::unique_future<void> res = pt.get_future();
	boost::thread task(boost::move(pt));
#else
	future<void> res = async(launch::async, &RandomForests::GenerateRF, random_forests, (int(m[0])),(bool(iW[0])), (int(par[0])), &interrupt);
#endif
	try {

		while (true) {
//			this_thread::sleep_for(chrono::seconds {1});

			// check interruption
			if (check_interrupt()) {
				interrupt = true;
				throw interrupt_exception("The random forest model building is interrupted.");
			}

			// check RF thread completion
#ifdef WSRF_USE_BOOST
			if (res.wait_for(boost::chrono::seconds (0)) == boost::future_status::ready) {
#else

#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 7
			if (res.wait_for(chrono::seconds {0}) == future_status::ready) {
#else
			if (res.wait_for(chrono::seconds {0})) {
#endif
#endif
				res.get();
				break;
			}
		}

	} catch (...) {

		// make sure thread is finished
		if (res.valid())
			res.get();

//		// cleanup
//		delete random_forests;
//		delete training_set;
#ifdef WSRF_USE_BOOST
		boost::rethrow_exception(boost::current_exception());
#else
		rethrow_exception(current_exception());
#endif


	}

#else
	volatile bool interrupt = false;
	vector<vector<int> > training_sets = random_forests->GetRandomTrainingSet();
	int ind = 0;
	for (vector<vector<int> >::iterator iter = training_sets.begin(); iter != training_sets.end(); ++iter, ind++) {

		// check interruption
		if (check_interrupt())
			throw interrupt_exception("The random forest model building is interrupted.");

		DecisionTree* decision_tree = new DecisionTree(training_set, sds[ind]);
		Node* root = decision_tree->GenerateDecisionTreeByC4_5(*iter, training_set->GetNormalAttributes(), int(m[0]), bool(iW[0]), &interrupt);
		decision_tree->set_root_(root);
		decision_tree->setOOBErrorRate(decision_tree->GetErrorRate(training_set, random_forests->getOOBset(ind)));
		random_forests->random_forests_.push_back(decision_tree);

	}

#endif

	if (!(bool(iPt[0]))) {
		RFDeployment RF_deployment(random_forests);
		RF_deployment.CalculateOOBPredictor(training_set);
		RF_deployment.CalculateOOBPredictorResult(training_set);
		RF_deployment.CalculateOOBConfusionMatrix(training_set);
		RF_deployment.CalculateOOBErrorRate(training_set);
		random_forests->set_OOB_error_rate_(RF_deployment.get_OOB_error_rate_());
		RF_deployment.CalculateRFStrength(training_set);
		RF_deployment.CalculateRFCorrelation(training_set);
		double s = RF_deployment.get_RF_strength_();
		random_forests->set_strength_(s);
		double c = RF_deployment.get_RF_correlation();
		random_forests->set_correlation_(c);
		random_forests->set_c_s2_(c / (s * s));
	}
	
	/*
	 *using formula,attribute_value_mapper.save_name_data() needn't be used
	 */
	Rcpp::List random_forests_R;
	if (!(bool(iPt[0]))) {
		random_forests_R["names"] = attribute_value_mapper->save_name_data();
		random_forests_R["ntrees"] = random_forests->get_trees_num();
	}
	random_forests_R["model"] = random_forests->save_model((bool(iPt[0])));

	delete random_forests;
	delete training_set;
	delete attribute_value_mapper;

	return random_forests_R;

	END_RCPP
}

SEXP merge(SEXP rfA, SEXP rfB)
{
	BEGIN_RCPP

	RandomForests * randomForestsA = new RandomForests(rfA, true);
	RandomForests * randomForestsB = new RandomForests(rfB, true);
	RandomForests * mergeRF = new RandomForests();
	int num_treeA = randomForestsA->get_trees_num();
	int num_treeB = randomForestsB->get_trees_num();
	mergeRF->set_trees_num(num_treeA + num_treeB);

	for(int i = 0; i < num_treeA; i++) {
		mergeRF->random_forests_.push_back(randomForestsA->random_forests_.at(i));
		mergeRF->selected_status_.push_back(randomForestsA->selected_status_.at(i));
	}
	randomForestsA->random_forests_.clear();

	for(int i = 0; i < num_treeB; i++) {
		mergeRF->random_forests_.push_back(randomForestsB->random_forests_.at(i));
		mergeRF->selected_status_.push_back(randomForestsB->selected_status_.at(i));
	}
	randomForestsB->random_forests_.clear();

	Rcpp::List random_forests_R;
	random_forests_R["model"] = mergeRF->save_model(true);

	delete randomForestsA;
	delete randomForestsB;
	delete mergeRF;

	return random_forests_R;

	END_RCPP
}

SEXP afterMerge(SEXP wrf, SEXP ds, SEXP nm)
{
	BEGIN_RCPP

	Rcpp::DataFrame ds_df = Rcpp::DataFrame(ds);
	Rcpp::DataFrame nm_df = Rcpp::DataFrame(nm);

	AttributeValueMapper* attribute_value_mapper = new AttributeValueMapper();
	attribute_value_mapper->MapAttributeValueRcpp(nm_df);

	TrainingSet* training_set = new TrainingSet();
	training_set->set_attribute_value_mapper(attribute_value_mapper);
	if(training_set->ProduceTrainingSetMatrixRcpp(ds_df) == FAIL){
		throw std::range_error("can't produce the training set !");
	}

	RandomForests* random_forests = new RandomForests(wrf, true);

	RFDeployment RF_deployment(random_forests);
	RF_deployment.CalculateOOBPredictor(training_set);
	RF_deployment.CalculateOOBPredictorResult(training_set);
	RF_deployment.CalculateOOBConfusionMatrix(training_set);
	RF_deployment.CalculateOOBErrorRate(training_set);
	random_forests->set_OOB_error_rate_(RF_deployment.get_OOB_error_rate_());
	RF_deployment.CalculateRFStrength(training_set);
	RF_deployment.CalculateRFCorrelation(training_set);
	double s = RF_deployment.get_RF_strength_();
	random_forests->set_strength_(s);
	double c = RF_deployment.get_RF_correlation();
	random_forests->set_correlation_(c);
	random_forests->set_c_s2_(c / (s * s));

	Rcpp::List random_forests_R;
	random_forests_R["names"] = attribute_value_mapper->save_name_data();
	random_forests_R["model"] = random_forests->save_model();
	random_forests_R["ntrees"] = random_forests->get_trees_num();

	delete training_set;
	delete random_forests;
	delete attribute_value_mapper;

	return random_forests_R;


	END_RCPP
}

SEXP predict(SEXP wrf, SEXP ds, SEXP type){

	BEGIN_RCPP

	Rcpp::DataFrame ds_df = Rcpp::DataFrame(ds);

	Rcpp::List wrf_df = Rcpp::List(wrf);
	Rcpp::DataFrame nm_df = Rcpp::DataFrame(SEXP(wrf_df["names"]));
	AttributeValueMapper* attribute_value_mapper = new AttributeValueMapper();
	attribute_value_mapper->MapAttributeValueRcpp(nm_df);
	
	TrainingSet* training_set = new TrainingSet();
	training_set->set_attribute_value_mapper(attribute_value_mapper);
    if(training_set->ProduceTrainingSetMatrixRcpp(ds_df, true) == FAIL){
    	throw std::range_error("can't produce the training set !");
	}
	
	RandomForests random_forests(wrf);

	string tp = Rcpp::as<string>(type);
	if (tp == "aprob"){
		vector< map<string, double> > predict_result = random_forests.PredictAprobMatrix(training_set, attribute_value_mapper);
		delete attribute_value_mapper;
		return Rcpp::wrap(predict_result);
	} else if (tp == "waprob") {
		vector< map<string, double> > predict_result = random_forests.PredictWaprobMatrix(training_set, attribute_value_mapper);
		delete attribute_value_mapper;
		return Rcpp::wrap(predict_result);
	} else {
		vector< map<string, int> > predict_result = random_forests.PredictClassMatrix(training_set, attribute_value_mapper);
		delete attribute_value_mapper;
		return Rcpp::wrap(predict_result);
	}

	END_RCPP
}

void ToCommonOptions(const string& options, int& argc, char**& argv){
	vector<string> splited_result;
	splited_result.push_back("WeightedRandomForest");
	int size = options.size();
	string temp = "";
	int i;
	for (i = 0; i < size; ++ i) {
		if (options.at(i) != ' ') {
			temp += options.at(i);
		} else {
			splited_result.push_back(temp);
			temp = "";
		}
	}
	splited_result.push_back(temp);
	argc =splited_result.size();
	argv = (char**)malloc(sizeof(char**) * argc);

	for(i = 0; i < argc; ++ i){
		argv[i] = (char*)malloc(sizeof(char) * (splited_result.at(i).size() + 1));
		splited_result.at(i).copy(argv[i], splited_result.at(i).size());
		argv[i][splited_result.at(i).size()] = '\0';

	}

}

/*
SEXP varWeights(SEXP dataSEXP, SEXP inputsSEXP, SEXP targetSEXP, SEXP typeSEXP) {

	BEGIN_RCPP

	Rcpp::NumericMatrix data = Rcpp::as<Rcpp::NumericMatrix>(dataSEXP);
	Rcpp::IntegerVector inputs = Rcpp::as<Rcpp::IntegerVector>(inputsSEXP) - 1;
	Rcpp::IntegerVector target = Rcpp::as<Rcpp::IntegerVector>(targetSEXP) - 1;
	Rcpp::CharacterVector type = Rcpp::as<Rcpp::CharacterVector>(typeSEXP);

	Rcpp::NumericVector values;

	if (type[0] == "cor") {
		values = pearsonCorrelation(data, inputs, target[0]);

	} else if (type[0] == "chisq"){
		values = chiSquared(data, inputs, target[0]);
	}

	return values / Rcpp::sum(values);

	END_RCPP

}

Rcpp::NumericVector pearsonCorrelation(Rcpp::NumericMatrix& data, Rcpp::IntegerVector& inputs, int target) {
	int nrows = data.nrows;
	int ncols = inputs.size();
	Rcpp::NumericVector corrs = Rcpp::NumericVector(ncols);
	Rcpp::NumericVector sumx = Rcpp::NumericVector(ncols, 0);
	Rcpp::NumericVector xx = Rcpp::NumericVector(ncols, 0);
	Rcpp::NumericVector xy = Rcpp::NumericVector(ncols, 0);
	double sumy = 0;
	double yy = 0;

	for (int row = 0; row < nrows; row++) {
		double y = data(row, target);
		sumy += y;
		yy += y * y;

		for (int col = 0; col < ncols; col++) {
			double x = data(row, inputs[col]);
			sumx[col] += x;
			xx[col] += x * x;
			xy[col] += x * y;
		}
	}

	for (int col = 0; col < ncols; col++) {
		double covxy = nrows * xy[col] - sumx[col] * sumy[col];
		double devx = nrows * xx[col] - sumx[col] * sumx[col];
		double devy = nrows * yy[col] - sumy[col] * sumy[col];
		double stdevxy = std::sqrt(devx * devy);
		corrs[col] = covxy / stdevxy;
	}

	return corrs;

}

Rcpp::NumericVector chiSquared(Rcpp::NumericMatrix& data, Rcpp::IntegerVector& inputs, int target) {
	int nrows = data.nrows;
	int ncols = inputs.size();
	Rcpp::NumericVector corrs = Rcpp::NumericVector(ncols);
	return corrs;
}
*/



