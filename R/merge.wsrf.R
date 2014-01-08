.merge.wsrf <- function(x, y, ...)
{
  return(.Call("merge", x, y, PACKAGE="wsrf"))
}      
