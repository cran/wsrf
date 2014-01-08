/*
 * utility.h
 *
 *  Created on: 28 Dec, 2011
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

/*the declaration in the file is used as the utility of the project                */

#ifndef UTILITY_H_
#define UTILITY_H_
#include <Rcpp.h>
#include<vector>
#include<string>
#include<map>

using namespace std;
/* enum StatusCode usually is used as the function return value type*/
enum StatusCode {SUCCESS,FAIL};

/*the union is used as the the item's attribute of the training sets,the arribute is
  a dsicrete value or a continuous value*/
enum ItemType{DISCRETE,CONTINUOUS};
typedef ItemType AttributeType;

//enum NodeType{INTERNALNODE,LEAFNODE,UNKNOWN};
enum NodeType{LEAFNODE,INTERNALNODE,UNKNOWN};

typedef struct item_struct{
	union item_value{
		 short discrete_value_;  
         double continuous_value_;
	} value_; 
	ItemType type_;
}Item;

typedef vector<vector<string> > StringMatrix;

typedef union attribute_value {
		int discrete_value_;
		double continuous_value_;
}AttributeValue;

typedef struct attribute_selection_result{
	//AttributeType attribute_type_; // modified in 2/15/2012,TrainingSet�����п��Եõ�����
	int attribute_;
	double split_value_;
	map<int,vector<int> > splited_training_set;
	double info_gain_;
} AttributeSelectionResult;

typedef struct index_value {
		int index_;
		double value_;
} IndexValue;

typedef struct index_class {
	int index_;
	int class_;
} IndexClass;

struct point {
	double x_;
	double y_;
};

class interrupt_exception : public std::exception {
public:
	interrupt_exception(std::string message): detailed_message(message) {};
	virtual ~interrupt_exception() throw() {};
	virtual const char* what() const throw() {
		return detailed_message.c_str();
	}
	std::string detailed_message;
};

static inline void check_interrupt_impl(void* /*dummy*/) {
    R_CheckUserInterrupt();
}

inline bool check_interrupt() {
	return (R_ToplevelExec(check_interrupt_impl, NULL) == FALSE);
}


#endif
