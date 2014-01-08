#ifndef _weightedRandomForest_C45_H
#define _weightedRandomForest_C45_H

#include <Rcpp.h>

/*
 * note : RcppExport is an alias to `extern "C"` defined by Rcpp.
 *
 * It gives C calling convention to the rcpp_hello_world function so that 
 * it can be called from .Call in R. Otherwise, the C++ compiler mangles the 
 * name of the function and .Call can't find it.
 *
 * It is only useful to use RcppExport when the function is intended to be called
 * by .Call. See the thread http://thread.gmane.org/gmane.comp.lang.r.rcpp/649/focus=672
 * on Rcpp-devel for a misuse of RcppExport
 */

RcppExport SEXP WeightedRandomForest(SEXP dsSEXP, SEXP nmSEXP, SEXP nTrees, SEXP nvars,SEXP isWeighted,SEXP parallel, SEXP seeds, SEXP isPart);
RcppExport SEXP predict(SEXP wrf, SEXP ds, SEXP aprob);
RcppExport SEXP merge(SEXP rfA, SEXP rfB);
RcppExport SEXP afterMerge(SEXP wrf, SEXP ds, SEXP nm);
#endif
