print.wsrf <- function(x, tree, ...)
{
  attrMeta   <- x$names
  attrValues <- strsplit(attrMeta$type, ",")
  attrNames  <- attrMeta$variable
    
  model      <- x$model
  trees      <- model$trees
  errRates   <- model$OOBErrorRatesForEachTree

  if (missing(tree))
  {
    for (i in 1:length(trees))
    {
      cat("\n")
      cat("Tree", i, "with", length(trees[[i]]), "nodes:\n")
      .printNode(c(1, ""), "root", 0, trees[[i]], attrNames, attrValues)
    }
  }
  else
  {
    nNodes <- length(trees[[tree]])
    cat("Tree ", tree, " has ", nNodes, " nodes with error rate ",
        round(errRates[tree], 3), ".\n", sep="")
    if (nNodes > 0)
      .printNode(c(1, ""), "root", 0, trees[[tree]], attrNames, attrValues)
  }
  invisible()
}

.printNode <- function(nodeIdValuePair, name, level, tree, attrNames, attrValues)
{
  LEAFNODE     <- 0
  INTERNALNODE <- 1
  DISCRETE     <- 0
  CONTINUOUS   <- 1
  INDENT       <- " .."
    
  id <- as.integer(nodeIdValuePair[1])
  value <- as.character(nodeIdValuePair[2])
  node <- tree[[id]]
  nodeType <- as.integer(node[2])
  
  if (nodeType == INTERNALNODE)
  {
    attr <- as.integer(node[3]) + 1
    attrType <- as.integer(node[4])
        
    # print current internal node info

    cat(rep(INDENT, level), id, ") ", name, value, "\n", sep="")
        
    # print child nodes
    if (attrType == DISCRETE)
    {
      nodeIdValuePairs <- cbind(node[-(1:4)], paste(" =", attrValues[[attr]]))
    }
    else if (attrType == CONTINUOUS)
    {
      splitValue <- node[5]
      nodeIdValuePairs <- cbind(node[-(1:5)], paste(c(" <=", " >"), splitValue))
    }
    else
      stop("Unknown attribute type!")
            
    apply(nodeIdValuePairs, 1, .printNode, attrNames[attr],
          level+1, tree, attrNames, attrValues)
        
  }
  else if (nodeType == LEAFNODE)
  {
    label <- node[3]
    labelName <- attrValues[[length(attrValues)-1]][label+1]
    classNums <- node[-(1:3)]
    probs <- round(classNums / sum(classNums), 3)
        
    # print current leaf node info
    
    cat(rep(INDENT, level), id, ") ", name, value, " ", labelName,
        " (", paste(probs, collapse=" "), ")", "\n", sep="")
        
  }
  else
    stop("Unknown node type!")
}



