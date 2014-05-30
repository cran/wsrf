oobErrorRate.wsrf <- function(object, tree)
{
    # trees=	a number indicates tree id of the tree to be summarised
    
    model      <- object$model
    ntrees     <- object$ntrees
    statistics <- model$estimation
    errRates   <- model$OOBErrorRatesForEachTree
    
    # return out-of-bag error rate for the forest, length of 1
    if (missing(tree) || (is.logical(tree) && !tree))
        return(statistics["OOBErrorRate"])
    
    # return out-of-bag error rates for specific trees
    if (is.integer(tree) && all(tree < ntrees))
        return(errRates[tree])
    
    # return out-of-bag error rates for all individual trees, length of ntrees
    if (is.logical(tree) && tree)
        return(errRates)
}
