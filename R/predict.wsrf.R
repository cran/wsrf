predict.wsrf <- function(object,
                         newdata,
                         type=c("response", "class", "prob", "vote", "aprob", "waprob"), ...)
{
  if (!inherits(object, "wsrf")) 
    stop("Not a legitimate wsrf object")

  if (missing(type)) type <- "class"
  if (type=="response") type <- "class"

  type <- match.arg(type)

  # The C code for predict does not handle missing values. So handle
  # them here by removing them from the dataset and then add in, in
  # the correct places, NA as the results from predict.

  complete <- complete.cases(newdata)
  rnames   <- rownames(newdata)
  newdata  <- newdata[complete,]
  
  # function "predict()" in C returns "votes" by default, 
  # and can also directly returns "aprob" or "waprob" correspondingly in terms of <type>
  # but "class" and "prob" will be treated as "votes",
  # so "class" and "prob" still need to be calculated in R below

  res <- .Call("predict", object, newdata, type, PACKAGE="wsrf")
  colnames(res) <- rownames(newdata)
  res <- t(res)

  # Deal with observations with missing values.
  
  nc  <- ncol(res)
  nr  <- sum(!complete)
  nas <- data.frame(matrix(rep(NA, nr*nc), ncol=nc), row.names=rnames[!complete])
  colnames(nas) <- colnames(res)
  fin <- rbind(res, nas)
  fin <- fin[order(as.integer(rownames(fin))),]
  
  if (type %in% c("aprob", "waprob"))
    return(fin)
  else if (type == "vote")
    return(fin)
  else if (type == "prob")
  {
    max.votes <- unique(apply(res, 1, sum))
    if(length(max.votes)!=1) stop("Differening number of votes found?")
    return(fin/max.votes)
  }
  else if (type == "class")
  {
    cl <- factor(rep(NA, length(complete)), levels=colnames(res))
    cl[complete] <- factor(colnames(res)[apply(res, 1, which.max)], levels=colnames(res))
    return(cl)
  }
}
