predict.wsrf <- function(object,
                         newdata,
                         type=c("response", "class", "prob", "vote", "aprob", "waprob"), ...)
{
  if (!inherits(object, "wsrf")) 
    stop("Not a legitimate wsrf object")

  if (missing(type)) type <- "class"
  if (type=="response") type <- "class"
  
  type <- match.arg(type)
  
  # function "predict()" in C returns "votes" by default, 
  # and can also directly returns "aprob" or "waprob" correspondingly in terms of <type>
  # but "class" and "prob" will be treated as "votes",
  # so "class" and "prob" still need to be calculated in R below
  
  res <- .Call("predict", object, na.fail(newdata[object$vars]), type, PACKAGE="wsrf")
  names(res) <- rownames(newdata)
  res <- do.call(rbind, res)
  
  if (type %in% c("aprob", "waprob"))
      return(res)
  else
      votes <- res

  # Check expected conditions

  max.votes <- unique(apply(votes, 1, sum))
  if(length(max.votes)!=1)
    stop("Differening number of votes found?")
  
  # Return the result.

  classes <- factor(colnames(votes)[apply(votes, 1, which.max)],
                    levels=colnames(votes))

  probs <- votes/max.votes
  
  if (type == "class")
    return(classes)
  else if (type == "prob")
    return(probs)
  else if (type == "vote")
    return(votes)

}

