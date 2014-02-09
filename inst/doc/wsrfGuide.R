
## @knitr setup, echo=FALSE, message=FALSE
library(knitr)
opts_chunk$set(tidy=FALSE)

opts_chunk$set(out.lines=4)
opts_chunk$set(out.truncate=80)

hook_output <- knit_hooks$get("output")
hook_source <- knit_hooks$get("source")
knit_hooks$set(output=function(x, options) 
{
  if (options$results != "asis")
  {
    # Split string into separate lines.
    x <- unlist(stringr::str_split(x, "\n"))
    # Trim to the number of lines specified.
    if (!is.null(n <- options$out.lines)) 
    {
      if (length(x) > n) 
      {
        # Truncate the output.
        x <- c(head(x, n), "....\n")
      }
    }
    # Truncate each line to length specified.
    if (!is.null(m <- options$out.truncate))
    {
      len <- nchar(x)
      x[len>m] <- paste0(substr(x[len>m], 0, m-3), "...")
    }
    # Paste lines back together.
    x <- paste(x, collapse="\n")
    # Replace ' = ' with '=' - my preference. Hopefully won't 
    # affect things inappropriately.
    x <- gsub(" = ", "=", x)
  }
  hook_output(x, options)
},
source=function(x, options)
{
  # Split string into separate lines.
  x <- unlist(stringr::str_split(x, "\n"))
  # Trim to the number of lines specified.
  if (!is.null(n <- options$src.top)) 
  {
    if (length(x) > n) 
    {
      # Truncate the output.
      if (is.null(m <-options$src.bot)) m <- 0
      x <- c(head(x, n+1), "\n....\n", tail(x, m+2)) 
   }
  }
  # Paste lines back together.
  x <- paste(x, collapse="\n")
  hook_source(x, options)
})



## @knitr eval=FALSE
## install.packages("wsrf")


## @knitr eval=FALSE
## install.packages("wsrf", configure.args="--enable-c11=yes")


## @knitr eval=FALSE
## install.packages("wsrf",
##                  configure.args="--enable-c11=no")


## @knitr eval=FALSE
## install.packages("wsrf",
##                  configure.args="--with-boost-include=<Boost include path>
##                                  --with-boost-lib=<Boost lib path>")


## @knitr usage_load, message=FALSE
library(rattle)
ds <- weather
dim(ds)
names(ds)


## @knitr usage_prepare
target <- "RainTomorrow"
id     <- c("Date", "Location")
risk   <- "RISK_MM"
ignore <- c(id, if (exists("risk")) risk)

(vars <- setdiff(names(ds), ignore))
dim(ds[vars])


## @knitr message=FALSE
library(randomForest)
if (sum(is.na(ds[vars]))) ds[vars] <- na.roughfix(ds[vars])
ds[target] <- as.factor(ds[[target]])
(tt <- table(ds[target]))


## @knitr 
(form <- as.formula(paste(target, "~ .")))


## @knitr 
seed <- 42
set.seed(seed)
length(train <- sample(nrow(ds), 0.7*nrow(ds)))
length(test <- setdiff(seq_len(nrow(ds)), train))


## @knitr eval=FALSE
## wsrf(formula,
##      data,
##      ntrees=500,
##      nvars=NULL,
##      weights=TRUE,
##      parallel=TRUE)


## @knitr usage_build_by_default, message=FALSE
library(wsrf)
model.wsrf <- wsrf(form, data=ds[train, vars])
print(model.wsrf, summary=TRUE)


## @knitr 
strength(model.wsrf)
correlation(model.wsrf)


## @knitr usage_evaluate
cl <- predict(model.wsrf, newdata=ds[test, vars], type="class")
actual <- ds[test, target]
(accuracy.wsrf <- sum(cl == actual, na.rm=TRUE)/length(actual))


## @knitr compare_with_cforest_randomForest, message=FALSE
library(randomForest)
library(party)
model.randomForest <- randomForest(form, data=ds[train, vars])
model.cforest <- cforest(form, data=ds[train, vars])

cl <- predict(model.randomForest, newdata=ds[test, vars], type="response")
actual <- ds[test, target]
(accuracy.randomForest <- sum(cl == actual, na.rm=TRUE)/length(actual))

cl <- predict(model.cforest, newdata=ds[test, vars], type="response")
actual <- ds[test, target]
(accuracy.cforest <- sum(cl == actual, na.rm=TRUE)/length(actual))


## @knitr usage_build_on_cluster, eval=FALSE
## servers <- paste0("node", 31:40)
## model.wsrf <- wsrf(form, data=ds[train, vars], parallel=servers)


