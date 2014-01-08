suppressMessages(library(wsrf))
suppressMessages(library(rattle))
suppressMessages(library(randomForest))

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