correlation <- function(object, ...) UseMethod("correlation")

correlation.wsrf <- function(object, ...) {

    object$model$estimation["correlation"]

}