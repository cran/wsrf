\name{wsrf}

\alias{wsrf}

\concept{weighted subspace decision trees}
\concept{weighted subspace random forest}

\title{

  Build a Forest of Weighted Subspace Decision Trees

}

\description{
  
  Build weighted subspace decision trees to construct a forest.
  
}

\usage{

wsrf(formula, data, nvars, mtry, ntrees=500, weights=TRUE, 
                 parallel=TRUE, na.action=na.fail)

}

\arguments{
  
  \item{formula}{a formula, with a response but no interaction terms.}

  \item{data}{a data frame in which to interpret the variables named in
          the formula.}
  
  \item{ntrees}{number of trees to build on each server; By default, 500}
  
  \item{nvars, mtry}{number of variables to choose, with Breiman's
      default for random forests being the integer less than or equal
      to \eqn{log_2(ninputs) + 1}.  For compatibility with other
      \code{R} packages like \code{randomForest}, both \code{nvars}
      and \code{mtry} are supported, however, only one of them should
      be specified.}

  \item{weights}{logical.  \code{TRUE} for weighted subspace
      selection, which is the default; \code{FALSE} for random
      selection.}

  \item{na.action}{indicate the behaviour when encountering NA values
      in \code{data}.}
  
  \item{parallel}{whether to run multiple cores (TRUE), nodes, or
  sequentially (FALSE).}

}

\details{

  See Xu, Huang, Williams, Wang, and Ye (2012) for details

}

\value{
  
  An object of class \code{wsrf}.

}

\examples{
  \dontshow{
    library(wsrf)
    library(rattle)
    library(randomForest)
  }
  
  # prepare parameters
  ds <- get("weather")
  dim(ds)
  names(ds)
  target <- "RainTomorrow"
  id     <- c("Date", "Location")
  risk   <- "RISK_MM"
  ignore <- c(id, if (exists("risk")) risk) 
  vars   <- setdiff(names(ds), ignore)
  if (sum(is.na(ds[vars]))) ds[vars] <- na.roughfix(ds[vars])
  ds[target] <- as.factor(ds[[target]])
  (tt <- table(ds[target]))
  form <- as.formula(paste(target, "~ ."))
  set.seed(42)
  train <- sample(nrow(ds), 0.7*nrow(ds))
  test <- setdiff(seq_len(nrow(ds)), train)
  
  # build model
  model.wsrf.1 <- wsrf(form, data=ds[train, vars])
  
  # view model
  print(model.wsrf.1, tree=1)
  summary(model.wsrf.1)
  summary(model.wsrf.1, tree=c(1,500))
  
  # evaluate
  strength(model.wsrf.1)
  correlation(model.wsrf.1)
  cl <- predict(model.wsrf.1, newdata=ds[test, vars], type="response")
  actual <- ds[test, target]
  (accuracy.wsrf <- sum(cl==actual)/length(actual))
}

\references{
  Xu B, Huang JZ, Williams G, Wang Q, Ye Y (2012). "Classifying very
high-dimensional data with random forests built from small subspaces."
International Journal of Data Warehousing and Mining (IJDWM), 8(2),
44-63.
}

\author{
  
  He Zhao and Graham Williams (SIAT)
  
}

\keyword{ models }
\keyword{ classif }
