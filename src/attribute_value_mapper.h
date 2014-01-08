/*
 * attribute_value_mapper.h
 *
 *  Created on: 8 Feb, 2012
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#ifndef ATTRIBUTE_VALUE_MAPPER_H_
#define ATTRIBUTE_VALUE_MAPPER_H_

#include"utility.h"
#include<cstdio>
#include<cstdlib>
#include<map>
#include<string>
#include<iostream>
#include<fstream>
#include<sstream>
#include<Rcpp.h>
using namespace std;

/*
 * Variable meta data holder
 *
 * interprets raw meta data from file or Rcpp::DataFrame
 * provides interface for querying meta data
 */
class AttributeValueMapper{
private:
	string file_path;  // raw meta data file path
	int attribute_num_;  // total variable number
	vector<AttributeType> attribute_type_;  // vector indicating type of all attribute: discrete, or continuous
	map<string,int> attribute_name_mapper_;  // mapping table of <variable name> : <variable index>
	/*
	 * mapping table of discrete variable value
	 * <var index> : [ <valname 1> : <val 1>, <valname 2> : <val 2>, ...]
	 */
	map<int,map<string,int> > attribute_value_mapper_;
	int classify_attribute_;  // index of target variable
	Rcpp::CharacterVector variable_;  // raw meta data of variable name
	Rcpp::CharacterVector type_;  // raw meta data of variable type
public:
	AttributeValueMapper();
	/*
	 * raw meta data conversion functions
	 */
	StatusCode MapAttributeValueRcpp(const Rcpp::DataFrame& nm); // For SEXP data type
	AttributeValueMapper(string file_path);
	StatusCode split(const string &str,char delim,vector<string> &strVec);
	bool HaveASpace(const string& str);
	int GetAttributeNum();
	AttributeType GetAttributeType(int attribute);
	bool GetMapperValue(int attribute, string value,map<string,int>::iterator &it);
	int GetClassifyAttribute();
	int GetAttributeValueNum(int attribute);
	string GetAttributeName(int attribute);
	string GetAttributeValueName(int attribute,int index);
	StatusCode MapAttributeValue();  // for reading meta data from file
        void DeleteComment(string& str);
	void DeleteSymbol(string& str);
	void test();

	//save the name date to model to predict
	Rcpp::List save_name_data();
	
	

};
#endif
