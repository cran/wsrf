.onAttach <- function(libname, pkgname) {
    wsrfDescription <- "wsrf: An R Package for Scalable Weighted Subspace Random Forests."
    wsrfVersion <- read.dcf(file=system.file("DESCRIPTION", package=pkgname),
                            fields="Version")
    
    packageStartupMessage(wsrfDescription)
    packageStartupMessage(paste("Version", wsrfVersion))
#    packageStartupMessage("Type wsrfNews() to see new features/changes/bug fixes.")
}


wsrf <- function(formula,
                 data,
                 nvars,
                 mtry,
                 ntrees=500,
                 weights=TRUE,
                 parallel=TRUE, 
                 na.action=na.fail)
{
  # Determine the information provided by the formula.

  target <- as.character(formula[[2]]) # Assumes it is a two sided formula.
  inputs <- attr(terms.formula(formula, data=data), "term.labels")
  vars   <- union(inputs, target)

  # Retain just the dataset required, and perform the required
  # na.action, which defaults to faling if there is missing data in
  # the dataset.

  data <- as.data.frame(na.action(data[vars]))
  
  # For the C++ code, when nvars=-1 then nvars will be set to (log_2(n)
  # + 1). Rather than relying on the C++ default, we set the default
  # value here, making it more clearly accessible to the R users.
  #
  # For compatibility with the R package randomForest, both nvars and
  # mtry are supported, however, only one of them should be specified.
  # We noted that in rf.c of the package source for randomForest, they
  # also use nvar but set mtry to nvar!

  if (missing(nvars) && missing(mtry))
    nvars <- floor(log2(length(inputs))+1)
  else if(!missing(nvars) && !missing(mtry)) 
    stop("only either nvars or mtry can be specified")
  else if (missing(nvars))
    nvars <- mtry
  nvars <- floor(nvars)

  # Check for pre-conditions.
  
  if(! target %in% names(data))
    stop("The named target must be included in the dataset.")

  if (nvars > length(inputs))
    stop("The chosen number of variables is greater than actually available.")

  # Prepare to pass execution over to the suitable helper.
  
  nm    <- .get.names.info(data, target)
  seeds <- as.integer(runif(ntrees) * 10000000)
  
  # Determine what kind of parallel to perform. By default, when
  # parallel=TRUE, use 2 less than the number of cores available, or 1
  # core if there are only 2 cores.
	
  if (is.logical(parallel) || is.numeric(parallel))
  {
    if (is.logical(parallel) && parallel)
    {
      parallel <- detectCores()-2
      if (parallel < 1) parallel <- 1
    }
    model <- .wsrf(data, nm, ntrees, nvars, weights, parallel, seeds)
  }
  else if (is.vector(parallel))
  {
    model <- .clwsrf(data, nm, ntrees, nvars, weights, serverargs=parallel, seeds)
  }
  else 
    stop ("Parallel must be logical, character, or numeric.")
  
  class(model) <- "wsrf"

  return(model)
}

.wsrf <- function(data, nm, ntrees, nvars, weights, parallel, seeds, isPart=FALSE)
{
  model <- .Call("WeightedRandomForest", data, nm, ntrees, nvars,
                 weights, parallel, seeds, isPart, PACKAGE="wsrf")
  model$vars <- names(data)
  return(model)
}

.localwsrf <- function(serverargs, data, nm, nvars, weights)
{	
  ntrees <- serverargs[1][[1]]
  parallel <- serverargs[2][[1]]
  seeds <- serverargs[3][[1]]
  model <- .wsrf(data, nm, ntrees, nvars, weights, parallel, seeds, TRUE)
  return(model)
}

.clwsrf <- function(data, nm, ntrees, nvars, weights, serverargs, seeds)
{
  # Multiple cores on multiple servers.
  # where serverargs like c("apollo9", "apollo10", "apollo11", "apollo12")
  # or c(apollo9=5, apollo10=8, apollo11=-1)
  
  if (is.vector(serverargs, "character"))
  {
    nodes <- serverargs
    cl <- makeCluster(nodes)
    clusterEvalQ(cl, require(wsrf))
    parallels <- unlist(clusterCall(cl, function()
    {
      if (.Platform$OS.type == "windows") return(1)
      
      nthreads <- detectCores() - 2
      if (nthreads > 0)
        return(nthreads)
      else
        return(1)
    }))
  }
  else if (is.vector(serverargs, "numeric"))
  {
    nodes <- names(serverargs)
    cl <- makeCluster(nodes)
    clusterEvalQ(cl, require(wsrf))
    parallels <- unlist(clusterCall(cl, function()
    {
      if (.Platform$OS.type == "windows") return(1)
      
      nthreads <- detectCores() - 2
      if (nthreads > 0)
        return(nthreads)
      else
        return(1)
    }))
    parallels <- ifelse(serverargs > 0, serverargs, parallels)
  }
  else
    stop ("Parallel must be a vector of mode character/numeric.")
  
  nservers <- length(nodes)
    
  # just make sure each node has different RNGs in C code, time is
  # part of the seed, so this call won't make a reproducible result
    
  clusterSetRNGStream(cl)
    
  # follow specification in "serverargs", calculate corresponding tree
  # number for each node
    
  nTreesPerNode <- floor(ntrees / sum(parallels)) * parallels
  nTreesLeft <- ntrees %% sum(parallels)
  #    cumsumParallels <- cumsum(parallels)
  #    leftPerNode <- ifelse(nTreesLeft >= cumsumParallels, parallels, 0)
  #    if (!(nTreesLeft %in% cumsumParallels)) {
  #        index <- which(nTreesLeft < cumsumParallels)[1]
  #        if (index == 1)
  #            leftPerNode[index] <- nTreesLeft
  #        else
  #            leftPerNode[index] <- nTreesLeft - cumsumParallels[index - 1]
  #    }
  
  ones <- rep(1, length(parallels))
  leftPerNode <- floor(nTreesLeft / sum(ones)) * ones
  left <- nTreesLeft %% sum(ones)
  leftPerNode <- leftPerNode + c(rep(1, left), rep(0, length(parallels) - left))
  
  nTreesPerNode <- nTreesPerNode + leftPerNode
  
  parallels <- parallels[which(nTreesPerNode > 0)]
  parallels <- as.integer(parallels)
  nTreesPerNode <- nTreesPerNode[which(nTreesPerNode > 0)]
  nTreesPerNode <- as.integer(nTreesPerNode)
  
  seedsPerNode <- split(seeds, rep(1:nservers, nTreesPerNode))
  
  forests <- parRapply(cl, cbind(nTreesPerNode, parallels, seedsPerNode),
                       .localwsrf, data, nm, nvars, weights)
  result <- Reduce(.merge.wsrf, forests)
  stopCluster(cl)
  
  # "afterMerge" is used for calculating strength, etc.
  result <- .Call("afterMerge", result, data, nm, PACKAGE="wsrf")
  result$vars <- names(data)
  return(result)
}

#get.names.info(data,target) function
#create a data structure from data set as following for compatibility with the C++ code program
#and avoid changing the c++ program
#the following is an example of the data structure
#
# attributes.names              attributes.values
# 1                  V1                     CONTINUOUS
# 2                  V2                    FEMALE,MALE
# 3                  V3 INNER_CITY,RURAL,SUBURBAN,TOWN
# 4                  V4                     CONTINUOUS
# 5                  V5                     CONTINUOUS
# 6                  V6                         NO,YES
# 7                  V7                         NO,YES
# 8                  V8                         NO,YES
# 9                  V9                         NO,YES
# 10 CLASSIFY_ATTRIBUTE                             V9

.get.names.info <- function(data, target)
{
  data[[target]] <- as.factor(data[[target]])
  attributes.names <- names(data)
  type <- sapply(data, class)	
  attributes.values <- character()
  index <- 1
  for(var in type)
  {
    attribute.value <- ""
    if("factor" %in% var)
    {
      attribute.values <- levels(data[[index]])
      attribute.values.num <- length(attribute.values)
      i <- 1
      for(var_value in attribute.values)
      {
        if(i < attribute.values.num)
        {
          attribute.value <- paste(attribute.value, var_value, ",", sep="")
        }
        else
        {
           attribute.value <- paste(attribute.value, var_value, sep="")
        }
        i <- i+1
      }
    }
    else
    {
      attribute.value <- "CONTINUOUS"
    }
    
    index <- index + 1
    attributes.values <- c(attributes.values, attribute.value)
  }
  attributes.names <- c(attributes.names, "CLASSIFY_ATTRIBUTE")
  attributes.values <- c(attributes.values, target)

  return(data.frame(attributes.names, attributes.values))
}
