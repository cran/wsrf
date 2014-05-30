\name{summary.wsrf}

\alias{summary}
\alias{summary.wsrf}

\title{
  Summarize a \code{wsrf} Model
}

\description{
  Summary method for class "\code{wsrf}".
}

\usage{
\method{summary}{wsrf}(object, tree, \dots)
}

\arguments{
  \item{object}{object of class \code{wsrf}.}
  \item{tree}{ the index of a specific tree in the forest model.  If
      provided, \code{summary.wsrf} will give the summary of the exact
      tree in the forest.  }
  \item{\dots}{Optional additional arguments. At present no additional
  arguments are used.}
}

\author{
  He Zhao and Graham Williams (SIAT)
}

\seealso{
  \code{\link{wsrf}}
}
