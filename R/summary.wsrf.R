summary.wsrf <- function(object, tree, ...)
{
  # trees=	a number indicates tree id of the tree to be summarised
  
  attrMeta   <- object$names
  attrValues <- strsplit(attrMeta$type, ",")
  attrNames  <- attrMeta$variable

  model      <- object$model
  statistics <- model$estimation
  tree.list  <- model$trees
  errRates   <- model$OOBErrorRatesForEachTree
  nTrees     <- length(tree.list)

  if (!missing(tree) && is.logical(tree) && tree) tree <- seq(tree.list)

  if (missing(tree) || is.logical(tree)) # If logical then will be FALSE now.
  {
    cat("A Weighted Subspace Random Forest model with ", nTrees,
        " tree", ifelse(nTrees == 1, "", "s"), ".\n\n", sep="")
        
    cat("Out-of-Bag Error Rate: ", statistics["OOBErrorRate"], "\n", sep="")
    cat("Strength: ", statistics["strength"], "\n", sep="")
    cat("Correlation: ", statistics["correlation"], "\n\n", sep="")
  }
  else
  {
    cat.tree.line <- function(n)
    {
      cat(sprintf("Tree %d has %d nodes with error rate %0.3f\n",
                  n, length(tree.list[[n]]), errRates[n], 3))
    }
    lapply(tree, cat.tree.line)
  }
  invisible()
}


