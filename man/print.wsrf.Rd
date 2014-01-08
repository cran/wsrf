\name{print.wsrf}

\alias{print}
\alias{print.wsrf}

\title{ Print Method for \code{wsrf} model }

\description{ Print all trees or one specific tree in the forest model
  built from \code{wsrf}.  }

\usage{
\method{print}{wsrf}(x, tree, \dots)
}

\arguments{
  \item{x}{object of class \code{wsrf}.}
  \item{tree}{the index of a specific tree.  If missing, \code{print}
      will print the whole forest, which will take a long time for a
      big forest.}
  \item{\dots}{other arguments to \code{print.wsrf}.}
}

\note{ It will take quite amount of time printing the whole forest if
  there are a large number of trees and a large number of nodes for
  each tree.}

\author{
  He Zhao and Graham Williams (SIAT)
}

\seealso{
  \code{\link{wsrf}}
}
