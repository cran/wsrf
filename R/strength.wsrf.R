strength <- function(object, ...) UseMethod("strength")

strength.wsrf <- function (object, ...) {

    object$model$estimation["strength"]

} 