/*
 * random_forests.cpp
 *
 *  Created on: 31 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */


/*
 * a.the random number generate?
 * b.is the training set repeatalbe?
 * c.the attribute can be repeatable
 */


#include"random_forests.h"

const string RandomForests::TREES = "trees";
const string RandomForests::SELECTED_STATUS = "selected_status";
const string RandomForests::ESTIMATION = "estimation";
const string RandomForests::OOB_ERROR_RATE = "OOBErrorRate";
const string RandomForests::STRENGTH = "strength";
const string RandomForests::CORRELATION = "correlation";
const string RandomForests::C_S2 = "c_s2";
const string RandomForests::OOB_ERROR_RATES_FOR_EACH_TREE = "OOBErrorRatesForEachTree";

RandomForests::RandomForests(){

}

/*
 * construct forest from R
 * parameter:
 *     isPart - indicate whether this rf is part of a larger distributed rf; mainly used in merge
 */
RandomForests::RandomForests(SEXP& rf, bool isPart){
	Rcpp::List rf_data_frame = Rcpp::List(rf);
	Rcpp::List model = rf_data_frame["model"];
	
	Rcpp::List trees = model[TREES];
	vector<double> errorRatesForEachTree = Rcpp::as<vector<double> >((SEXP)(model[OOB_ERROR_RATES_FOR_EACH_TREE]));
	
	this->trees_num_ = trees.size();
	Rcpp::List::iterator it;
	int i;
	for(i = 0, it = trees.begin(); it != trees.end(); ++ it, i++){
		DecisionTree* decision_tree = new DecisionTree(Rcpp::List(SEXP(*it)));
		decision_tree->setOOBErrorRate(errorRatesForEachTree[i]);
		this->random_forests_.push_back(decision_tree);
	}
	
	if (isPart) {
		Rcpp::List statuses = model[SELECTED_STATUS];
#ifdef WSRF_USE_C11
		for_each (statuses.begin(), statuses.end(), [&](SEXP it) { this->selected_status_.push_back(Rcpp::as<vector<bool> >(it)); });
#else
		for (int i = 0; i < statuses.size(); i++)
			this->selected_status_.push_back(Rcpp::as<vector<bool> >(statuses[i]));
#endif
	} else {
		Rcpp::NumericVector estimation = model[ESTIMATION];
		this->OOB_error_rate_ = estimation[OOB_ERROR_RATE];
		this->strength_ = estimation[STRENGTH];
		this->correlation_ = estimation[CORRELATION];
		this->c_s2_ = estimation[C_S2];
	}
}

RandomForests::RandomForests(TrainingSet* training_set,int trees_num, Rcpp::IntegerVector seeds){
	this->training_set_=training_set;
	this->trees_num_=trees_num;

	for (int i = 0; i < seeds.size(); i++)
		this->tree_seeds.push_back(seeds[i]);
}

RandomForests::~RandomForests() {
	if (random_forests_.size() > 0)
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
		BOOST_FOREACH (DecisionTree* tree, random_forests_) {
#else
		for (auto tree : random_forests_) {
#endif
			if (tree)
				delete tree;
		}
#else
		for (int i = 0; i < random_forests_.size(); i++) {
			if (random_forests_[i])
				delete random_forests_[i];
		}
#endif

}

vector<DecisionTree*> RandomForests::get_trees(){
	return this->random_forests_;
}

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
/**
 * Build RandomForests::trees_num_ decision trees
 *
 * forest building controller
 *
 * interrupt check points:
 * 1. check at the beginning of every tree building: RandomForests::GenerateRF()
 * 2. check at the beginning of every node building: DecisionTree::GenerateDecisionTreeByC4_5()
 * 3. check at the beginning of time consuming operation: C4_5AttributeSelectionMethod::HandleDiscreteAttribute() in C4_5AttributeSelectionMethod::ExecuteSelectionByIGR()
 */
void RandomForests::GenerateRF(int nvars, bool isWeighted, int parallel, 
			       volatile bool* pInterrupt)
{
	vector<vector<int> > training_sets=this->GetRandomTrainingSet();
	vector<vector<int> >::iterator iter;

#ifdef WSRF_USE_BOOST
	int nCoresMinusTwo = boost::thread::hardware_concurrency() - 2;
#else
	int nCoresMinusTwo = thread::hardware_concurrency() - 2;
#endif

	if (parallel == 0 || parallel == 1 || (parallel < 0 && nCoresMinusTwo == 1)) {
		// build trees sequentially
		int ind = 0;
		for (iter = training_sets.begin(); iter != training_sets.end(); ++iter, ind++) {

			if (*pInterrupt) {
				break;
			}

			DecisionTree* decision_tree = new DecisionTree(this->training_set_, this->tree_seeds[ind]);
			Node* root = decision_tree->GenerateDecisionTreeByC4_5(*iter,
					this->training_set_->GetNormalAttributes(), nvars,
					isWeighted, pInterrupt);
			decision_tree->set_root_(root);
			decision_tree->setOOBErrorRate(decision_tree->GetErrorRate(this->training_set_, OOBSet[ind]));
			this->random_forests_.push_back(decision_tree);

		}

	} else {
		// simultaneously build <nThreads> trees until <tree_num_> trees has been built
		int nThreads;
		if (parallel < 0) {
			// gcc 4.6 do not support std::thread::hardware_concurrency(), we fix the thread number to 10
			nThreads = nCoresMinusTwo > 1 ? nCoresMinusTwo : 10;
		} else {
			nThreads = parallel;
		}

		// using <nThreads> tree builder to build trees
		int index = 0;
#ifdef WSRF_USE_BOOST
		boost::unique_future<void> results[nThreads];
		boost::thread tasks[nThreads];
#else
		future<void> results[nThreads];
#endif
		this->random_forests_ = vector<DecisionTree*>(this->trees_num_);
		for (int i = 0; i < nThreads; i++)
#ifdef WSRF_USE_BOOST
		{
			boost::packaged_task<void> pt(boost::bind(&RandomForests::GenerateTree, this, &index, nvars, isWeighted, pInterrupt));
			results[i] = pt.get_future();
			tasks[i] = boost::thread(boost::move(pt));
		}
#else
			results[i] = async(launch::async, &RandomForests::GenerateTree, this, &index, nvars, isWeighted, pInterrupt);
#endif

		try {
			bool mark[nThreads];
			for (int i = 0; i < nThreads; i++) mark[i] = false;
			int i = 0;
			do {
				// check each tree builder's status till all are finished
				for (int j = 0; j < nThreads; j++) {
#ifdef WSRF_USE_BOOST
					if (mark[j] != true && results[j].valid() && results[j].wait_for(boost::chrono::seconds (0)) == boost::future_status::ready) {
#else
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 7
					if (mark[j] != true && results[j].valid() && results[j].wait_for(chrono::seconds {0}) == future_status::ready) {
#else
					if (mark[j] != true && results[j].valid() && results[j].wait_for(chrono::seconds {0})) {
#endif
#endif
						results[j].get();
						mark[j] = true;
						i++;
					}
				}
//				this_thread::sleep_for(chrono::seconds(1));
			} while (i < nThreads);
        } catch (...) {
        	(*pInterrupt) = true;  // if one tree builder throw a exception, set true to inform others
#ifdef WSRF_USE_BOOST
        	boost::rethrow_exception(boost::current_exception());
#else
        	rethrow_exception(current_exception());
#endif
        }

	}

}

/*
 * tree building function: pick a bagging set from bagging_set_ to build one tree a time until no set to fetch
 */
void RandomForests::GenerateTree(int* index, int nvars, bool isWeighted, volatile bool* pInterrupt)
{
	bool finished = false;
	int ind;

	while (!finished) {

		if (*pInterrupt)
			break;

#ifdef WSRF_USE_BOOST
		boost::unique_lock<boost::mutex> ulk(mut);
#else
		unique_lock<mutex> ulk(mut);
#endif
		if (*index < trees_num_) {
			ind = *index;
			(*index)++;
			ulk.unlock();
		} else {
			finished = true;
		}

		if (!finished) {
			DecisionTree* decision_tree = new DecisionTree(training_set_, this->tree_seeds[ind]);
			Node* root = decision_tree->GenerateDecisionTreeByC4_5(bagging_set_[ind], training_set_->GetNormalAttributes(), nvars, isWeighted, pInterrupt);
			decision_tree->set_root_(root);
			decision_tree->setOOBErrorRate(decision_tree->GetErrorRate(this->training_set_, OOBSet[ind]));

			ulk.lock();
			random_forests_[ind] = decision_tree;
		}
	}
}

#endif

/*
 * randomly pick <trees_num_> * <training_set_num_> instances for every tree
 * return a vector contains indexes of those instances
 */
vector<vector<int> > RandomForests::GetRandomTrainingSet(){
	int max = this->training_set_->get_training_set_num_();
	this->selected_status_ = vector<vector<bool> >(this->trees_num_,vector<bool>(max,false));
	vector<vector<int> > all_vec;
	vector<int> vec;
	int random_num;
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	boost::random::uniform_int_distribution<int> uid(0, max - 1);
	for(int i=0;i<this->trees_num_;++i){
		boost::random::mt19937 re(this->tree_seeds[i]);
#else
	uniform_int_distribution<int> uid{0, max - 1};
	for(int i=0;i<this->trees_num_;++i){
		default_random_engine re {this->tree_seeds[i]};
#endif
		for(int j=0;j<max;++j){
			random_num = uid(re);
			vec.push_back(random_num);
			this->selected_status_.at(i).at(random_num) = true;
		}
#else
	int i;
	int k = 0;
	for(i=0;i<this->trees_num_;++i){
		srand(this->tree_seeds[i]);
		int j;
		for(j=0;j<max;++j){
//			srand(unsigned(time(NULL)+k));
			random_num=rand() % max;
			vec.push_back(random_num);
			this->selected_status_.at(i).at(random_num) = true;
			k++;
		}
#endif
		/*
		 * in the bagging, the duplicated instance cat not be discarded
		 */
//		sort(vec.begin(),vec.end());
//		vector<int>::iterator it=unique(vec.begin(),vec.end());
//		vec.resize(it-vec.begin());
		all_vec.push_back(vec);
		vec.clear();

	}

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	BOOST_FOREACH (vector<bool>& inBags, selected_status_) {
#else
	for (vector<bool>& inBags : selected_status_) {
#endif
		vector<int> oob;
		for (int ind = 0; ind < int(inBags.size()); ind++)
			if (!inBags[ind])
				oob.push_back(ind);
		OOBSet.push_back(oob);
	}
#else
	for (int i = 0; i < selected_status_.size(); i++) {
		vector<int> oob;
		for (int ind = 0; ind < int(selected_status_[i].size()); ind++)
			if (!selected_status_[i][ind])
				oob.push_back(ind);
		OOBSet.push_back(oob);
	}
#endif

	this->bagging_set_ = all_vec;
	return all_vec;
}

/*
 * return the majority class label predicted by forest
 * for one instance
 */
int RandomForests::PredictClass(TrainingSet* training_set,int tuple){
	map<int,int> mapper;
	vector<DecisionTree*>::iterator iter;
	for(iter=this->random_forests_.begin();iter!=this->random_forests_.end();++iter){
		int target_class=((*iter)->PredictClass(training_set,tuple,(*iter)->get_root_()))->get_class_();
		if(mapper.find(target_class)==mapper.end()){
			mapper.insert(map<int,int>::value_type(target_class,1));
		}else{
			mapper[target_class]++;
		}

	}
	int max=0;
	int result = 0;
	map<int,int>::iterator map_iter;
	for(map_iter=mapper.begin();map_iter!=mapper.end();++map_iter){
		if(map_iter->second>=max){
			result=map_iter->first;
			max = map_iter->second;
		}
	}
	return result;
}

/*
 * return <class label> : <predict count>
 * when predict count == 0, no corresponding entry return
 */
map<int,int> RandomForests::predictClassMapForOneInstance(TrainingSet* training_set,int tuple)
{
	map<int,int> mapper;
	vector<DecisionTree*>::iterator iter;
	for(iter=this->random_forests_.begin();iter!=this->random_forests_.end();++iter){
		int target_class=((*iter)->PredictClass(training_set,tuple,(*iter)->get_root_()))->get_class_();
		if(mapper.find(target_class)==mapper.end()){
			mapper.insert(map<int,int>::value_type(target_class,1));
		}else{
			mapper[target_class]++;
		}

	}
	return mapper;
}

vector<double> RandomForests::predictAprobVecForOneInstance(TrainingSet* training_set,int tuple) {
	vector<double> aprobs;
	vector<DecisionTree*>::iterator iter;
	for(iter=this->random_forests_.begin();iter!=this->random_forests_.end();++iter){
		vector<double> classDistributions=((*iter)->PredictClass(training_set,tuple,(*iter)->get_root_()))->getClassDistributions();
		if (aprobs.size() == 0)
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
			BOOST_FOREACH (double dist, classDistributions) aprobs.push_back(dist);
#else
			for (double dist : classDistributions) aprobs.push_back(dist);
#endif
#else
			for (int i = 0; i < classDistributions.size(); i++)
				aprobs.push_back(classDistributions[i]);
#endif
		else
			for (int i = 0; i < int(aprobs.size()); i++) aprobs[i] += classDistributions[i];
	}

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	BOOST_FOREACH (double& i, aprobs) i /= random_forests_.size();
#else
	for (double& i : aprobs) i /= random_forests_.size();
#endif
#else
	for (int i = 0; i < aprobs.size(); i++) aprobs[i] /= random_forests_.size();
#endif

	return aprobs;
}

vector<double> RandomForests::predictWaprobVecForOneInstance(TrainingSet* training_set,int tuple) {
	vector<double> waprobs;
	vector<DecisionTree*>::iterator iter;
	double sumAccuracy = 0;
	for(iter=this->random_forests_.begin();iter!=this->random_forests_.end();++iter){
		vector<double> classDistributions=((*iter)->PredictClass(training_set,tuple,(*iter)->get_root_()))->getClassDistributions();
		double accuracy = 1 - (*iter)->getOOBErrorRate();
		sumAccuracy += accuracy;

		if (waprobs.size() == 0)
#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
			BOOST_FOREACH (double dist, classDistributions) waprobs.push_back(dist * accuracy);
#else
			for (double dist : classDistributions) waprobs.push_back(dist * accuracy);
#endif
#else
			for (int i = 0; i < classDistributions.size(); i++)
				waprobs.push_back(classDistributions[i] * accuracy);
#endif
		else
			for (int i = 0; i < int(waprobs.size()); i++) waprobs[i] += classDistributions[i] * accuracy;
	}

#if defined WSRF_USE_BOOST || defined WSRF_USE_C11
#ifdef WSRF_USE_BOOST
	BOOST_FOREACH (double& i, waprobs) i /= sumAccuracy;
#else
	for (double& i : waprobs) i /= sumAccuracy;
#endif
#else
	for (int i = 0; i < waprobs.size(); i++) waprobs[i] /= sumAccuracy;
#endif

	return waprobs;
}

double RandomForests::GetErrorRate(TrainingSet* training_set,vector<int> training_set_index){
	int error=0;
	AttributeValue** value_matrix = training_set->GetValueMatrixP();
	int target_attribute = training_set->GetClassifyAttribute();
	vector<int>::iterator iter;
	for(iter=training_set_index.begin();iter!=training_set_index.end();++iter){
		if(this->PredictClass(training_set,*iter) != value_matrix[target_attribute][*iter].discrete_value_){
			error++;
		}
	}
	return ((double)error)/((double)training_set_index.size());
}

vector<vector<int> > RandomForests::get_bagging_set(){
	return this->bagging_set_;
}

int RandomForests::get_trees_num(){
	return this->trees_num_;
}

void RandomForests::set_trees_num(int num) {
	this->trees_num_ = num;
}

vector<vector<bool> > RandomForests::get_selected_status_(){
	return this->selected_status_;
}


void RandomForests::set_OOB_error_rate_(double rate){
	this->OOB_error_rate_ = rate;
}
double RandomForests::get_OOB_error_rate_(){
	return this->OOB_error_rate_;
}
void RandomForests::set_strength_(double strength){
	this->strength_ = strength;
}
double RandomForests::get_strength(){
	return this->strength_;
}
void RandomForests::set_correlation_(double correlation){
	this->correlation_ = correlation;
}
double RandomForests::get_correlation_(){
	return this->correlation_;
}
void RandomForests::set_c_s2_(double c_s2){
	this->c_s2_ = c_s2;
}
double RandomForests::get_c_s2_(){
	return this->c_s2_;
}

/*
 * output predict result into file
 * one line for one instance
 */
void RandomForests::PredictClassToFile(TrainingSet* training_set,
		               AttributeValueMapper* attribute_value_mapper,
		               string file_name){
	ofstream out(file_name.c_str());
	int case_num = training_set->get_training_set_num_();
	int target_attribute = attribute_value_mapper->GetClassifyAttribute();
	int i;
	int lable;
	string result;
	for(i = 0; i < case_num; ++ i){
		lable = this->PredictClass(training_set,i);
		result = attribute_value_mapper->GetAttributeValueName(target_attribute,lable);
		out << result << "\n";
	}
	out.close();
}

/*
 * return all predict results (include those with 0 votes)
 * for all instances in training_set
 */
vector< map<string, int> > RandomForests::PredictClassMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper){
	int case_num = training_set->get_training_set_num_();
	vector< map<string, int> > result_lable;
	int target_attribute = attribute_value_mapper->GetClassifyAttribute();
	int target_Num = attribute_value_mapper->GetAttributeValueNum(target_attribute);
	int i;
	for(i = 0; i < case_num; ++ i){
		map<int,int> mapper = this->predictClassMapForOneInstance(training_set,i);

		for(int target_ind = 0; target_ind < target_Num; target_ind++)
		{
			if(mapper.find(target_ind)==mapper.end()){
				mapper.insert(map<int,int>::value_type(target_ind,0));
			}
		}

		map<int,int>::iterator map_iter;
		//int size = mapper.size();
		int ind = 0;
		map<string, int> labelVotes;
		for(map_iter=mapper.begin();map_iter!=mapper.end();++map_iter, ++ind){
			int label = map_iter->first;
			string labelName = attribute_value_mapper->GetAttributeValueName(target_attribute,label);
			int votes = map_iter->second;
			labelVotes.insert(map<string,int>::value_type(labelName,votes));
		}
		result_lable.push_back(labelVotes);
	}
	return result_lable;
	
}

vector< map<string, double> > RandomForests::PredictAprobMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper) {

	int case_num = training_set->get_training_set_num_();
	vector< map<string, double> > result_lable;
	int target_attribute = attribute_value_mapper->GetClassifyAttribute();
	int target_Num = attribute_value_mapper->GetAttributeValueNum(target_attribute);
	for(int i = 0; i < case_num; ++ i){
		vector<double> aprob = this->predictAprobVecForOneInstance(training_set,i);

		map<string, double> labelProbs;
		for(int label = 0; label < target_Num; label++){
			string labelName = attribute_value_mapper->GetAttributeValueName(target_attribute,label);
			double prob = aprob[label];
			labelProbs.insert(map<string,double>::value_type(labelName,prob));
		}
		result_lable.push_back(labelProbs);
	}
	return result_lable;

}

vector< map<string, double> > RandomForests::PredictWaprobMatrix(TrainingSet* training_set, AttributeValueMapper* attribute_value_mapper) {

	int case_num = training_set->get_training_set_num_();
	vector< map<string, double> > result_lable;
	int target_attribute = attribute_value_mapper->GetClassifyAttribute();
	int target_Num = attribute_value_mapper->GetAttributeValueNum(target_attribute);
	for(int i = 0; i < case_num; ++ i){
		vector<double> aprob = this->predictWaprobVecForOneInstance(training_set,i);

		map<string, double> labelProbs;
		for(int label = 0; label < target_Num; label++){
			string labelName = attribute_value_mapper->GetAttributeValueName(target_attribute,label);
			double prob = aprob[label];
			labelProbs.insert(map<string,double>::value_type(labelName,prob));
		}
		result_lable.push_back(labelProbs);
	}
	return result_lable;

}

/*
 * wrap the forest model into R
 */
Rcpp::List RandomForests::save_model(bool isPart){
	
	Rcpp::List random_forests;
	vector<DecisionTree*>::iterator it;
	vector<vector<vector<double> > > trees;
	vector<double> errorRatesForEachTree;
	for(it = this->random_forests_.begin(); it != this->random_forests_.end(); ++ it){
		vector<vector<double> > tree;
		this->save_one_tree((*it)->get_root_(), tree);
		trees.push_back(tree);
		errorRatesForEachTree.push_back((*it)->getOOBErrorRate());
	
	}
	Rcpp::List trees_list = Rcpp::List(Rcpp::wrap(trees));
	random_forests[TREES] = trees_list;
	random_forests[OOB_ERROR_RATES_FOR_EACH_TREE] = Rcpp::wrap(errorRatesForEachTree);

	if (isPart) {
		random_forests[SELECTED_STATUS] = Rcpp::List(Rcpp::wrap(selected_status_));
	} else {
		Rcpp::NumericVector estimation;
		estimation[OOB_ERROR_RATE] = this->OOB_error_rate_;
		estimation[STRENGTH] = this->strength_;
		estimation[CORRELATION] = this->correlation_;
		estimation[C_S2] = this->c_s2_;
		random_forests[ESTIMATION] = estimation;
	}

	return random_forests;

}

/*
 * serialize a tree, which take <root> as root node, into <tree>
 * a serialized node of a tree is of type vector<double>,
 * so a serialzied tree is of type vector< vector<double> >,
 * and the serialization order of child nodes is from left to right
 *
 * see also DecisionTree::DecisionTree(Rcpp::List nodes_info)
 */
void RandomForests::save_one_tree(Node* root, vector<vector<double> >& tree){
	if (root == 0) return;

	vector<double> node_info = root->get_node_info();
	tree.push_back(node_info);
	if(root->GetNodeType() == INTERNALNODE){
		vector<Node*> childs = root->get_all_child_nodes_();
	        vector<Node*>::iterator it;
	        for(it = childs.begin(); it != childs.end(); ++ it){
		    this->save_one_tree(*it, tree);
	        }
	}else{
		return;
	}
}





