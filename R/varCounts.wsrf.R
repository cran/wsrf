varCounts.wsrf <- function(object) {
	# Return the times of each variable being selected as split condition.
	# For evaluating the bias of wsrf towards attribute types (categorical and
	# numerical) and the number of values each attribute has.

	attrMeta <- object$names
	model <- object$model
	
	attrNames <- attrMeta$variable
	trees <- model$trees
	
	LEAFNODE <- 0
	INTERNALNODE <- 1
	
	nTrees <- length(trees)
	nAttrs <- length(attrNames)
	counts <- vector("integer", nAttrs)
	names(counts) <- attrNames
	
	for (i in 1:nTrees) {
		tree <- trees[[i]]
		nNodes <- length(tree)
		
		for (j in 1:nNodes) {
			node <- tree[[j]]
			nodeType <- as.integer(node[2])
			
			if (nodeType == INTERNALNODE) {
				attr <- as.integer(node[3]) + 1
				counts[attr] <- counts[attr] + 1
			}
		}
	}
	
	return(counts)
}
