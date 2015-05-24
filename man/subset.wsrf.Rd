\name{subset.wsrf}
\alias{subset}
\alias{subset.wsrf}

\title{Subset of a Forest}

\description{
Obtain a subset of a forest.
}

\usage{
\method{subset}{wsrf}(x, trees, ...)
}

\arguments{
  \item{x}{an object of class \code{wsrf}.}
  \item{trees}{Which trees should be included in the sub-forest.
  An integer vector, which indicates the index of the trees.}
  \item{...}{not used.}
}

\value{
An object of class \code{wsrf}.
}

\seealso{\code{\link{combine}}}