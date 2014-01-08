/*
 * attribute_value_mapper.cpp
 *
 *  Created on: 8 Feb, 2012
 *      Author: meng
 *       email: qinghan.meng@gmail.com
 */

#include"attribute_value_mapper.h"

AttributeValueMapper::AttributeValueMapper(){

}

/*
 * convert raw meta data from Rcpp::DataFrame into
 * structured internal meta data
 */
StatusCode AttributeValueMapper::MapAttributeValueRcpp(const Rcpp::DataFrame& nm){
	// nm[0] may be name list of variables
	this->variable_ = nm[0];
	vector<string> variable_vec = Rcpp::as< vector<string> >(SEXP(this->variable_));

	// nm[1] may be type list of varialbes
	this->type_ = nm[1];
	vector<string> type_vec = Rcpp::as< vector<string> >(SEXP(this->type_));

	int variable_num = variable_vec.size();
	int type_num = type_vec.size();
	if(variable_num != type_num)
		throw std::range_error("the names file is wrong variable num does not equal type num");

	int attribute_num = 0;  // counts number of variables
	for(int i = 0;i < type_num;i++){
		string str_variable = variable_vec.at(i);  // current variable name
		string str_type = type_vec.at(i);  // current variable type

		// skip empty lines
		if( str_variable.empty() || str_type.empty() )
			continue;

		// line with prefix "#" may be comments
		if(str_variable.at(0) == '#')
			continue;

		if( str_variable == "CLASSIFY_ATTRIBUTE" ){
			this->classify_attribute_=this->attribute_name_mapper_[str_type];
			break;
		}else if( str_type == "CONTINUOUS" ){
			// variable of type continuous
			this->attribute_name_mapper_.insert(map<string,int>::value_type(str_variable,attribute_num));
			this->attribute_type_.push_back(CONTINUOUS);
		}else{
			// variable of type discrete
			this->attribute_name_mapper_.insert(map<string,int>::value_type(str_variable,attribute_num));
			this->attribute_type_.push_back(DISCRETE);

			// maps discrete variable value name and its value
			vector<string> temp_str_vec2;
			this->split(str_type,',',temp_str_vec2);
			map<string,int> temp_map;
			for(int j=0;j<int(temp_str_vec2.size());++j)
				temp_map.insert(map<string,int>::value_type(temp_str_vec2.at(j),j));
			this->attribute_value_mapper_.insert(map<int,map<string,int> >::value_type(attribute_num,temp_map));
		}

		attribute_num++;
	}
	this->attribute_num_=attribute_num;
	return SUCCESS;
}

AttributeValueMapper::AttributeValueMapper(string file_path){
	this->file_path=file_path;
}

/*
 * remove ' ', '.', '\'', '\t', '\r' from str
 */
void AttributeValueMapper::DeleteSymbol(string& str){
	int i;
    string temp_str="";
	for(i = 0;i < int(str.length());++ i){
		if(str[i] != ' ' && str [i] != '.' && str[i] != '\'' && str[i] != '\t' && str[i] != '\r'){
           temp_str += str[i];
		}
	}
	str.clear();
	str = temp_str;
}

/*
 * remove comment start with '|' in str
 */
void AttributeValueMapper::DeleteComment(string& str){
	size_t pos = str.find('|');
	if(string::npos == pos){
		return;
	}else{
		str= str.substr(0,pos);
	}

}

/*
 * separate str into parts with delm as separator
 */
StatusCode AttributeValueMapper::split(const string &str,char delim,vector<string> &strVec){
	string tempWord;
	stringstream tempSS(str);
	while(getline(tempSS,tempWord,delim)){
		strVec.push_back(tempWord);
	}
	return SUCCESS;

}

int AttributeValueMapper::GetAttributeNum(){
	return this->attribute_num_;
}

AttributeType AttributeValueMapper::GetAttributeType(int attribute){
	return this->attribute_type_.at(attribute);
}

//int AttributeValueMapper::GetMapperValue(int attribute,string value){
//	return this->attribute_value_mapper_[attribute][value];
//}
/*
 * return whether a discrete variable <attribute> permits a value <value>
 * and <it> is assigned the corresponding discrete value of that variable
 */
bool AttributeValueMapper::GetMapperValue(int attribute, string value, map<string,int>::iterator &it){
	it = this->attribute_value_mapper_[attribute].find(value);
	if (it == this->attribute_value_mapper_[attribute].end() ){
		return false;
	}else{
		return true;
	}
}

/*
 * return target variable index
 */
int AttributeValueMapper::GetClassifyAttribute(){
	return this->classify_attribute_;
}

/*
 * return possible value number of discrete variable <attribute>
 */
int AttributeValueMapper::GetAttributeValueNum(int attribute){
	return this->attribute_value_mapper_[attribute].size();
}

/*
 * return name of the value <index> of variable <attribute>
 */
string AttributeValueMapper::GetAttributeName(int attribute){
	if(attribute>=this->GetAttributeNum()){
		throw std::range_error("error in attribute_value_mapper.cpp in function GetAttributeName()");
	}
	map<string,int>::iterator iter=this->attribute_name_mapper_.begin();
	while(iter->second!=attribute){
		iter++;
	}
	return iter->first;
}

string AttributeValueMapper::GetAttributeValueName(int attribute,int index){
	if((attribute>=this->GetAttributeNum())||(index>=this->GetAttributeValueNum(attribute))){
		throw std::range_error("error in attribute_value_mapper.cpp in function GetAttributeValueName()");
	}
	map<string,int> mapper=this->attribute_value_mapper_[attribute];
	map<string,int>::iterator iter=mapper.begin();
	while(iter->second!=index){
		iter++;
	}
	return iter->first;
}

/*
 * return raw data back to R using Rcpp
 */
Rcpp::List AttributeValueMapper::save_name_data(){
	Rcpp::List name_data;
	name_data["variable"] = this->variable_;
	name_data["type"] = this->type_;
	return name_data;	
}
