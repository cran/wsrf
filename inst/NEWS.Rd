\name{NEWS}
\title{News for Package 'wsrf'}

\newcommand{\cpkg}{\href{http://CRAN.R-project.org/package=#1}{\pkg{#1}}}

\section{Changes in wsrf version 1.3.0 (2014-02-08)}{
  \itemize{

    \item added documents:
    \itemize{
      \item vignettes
      \item README.Rd
      \item NEWS.md
    }

    \item provided Windows support (no multi-threading functionality,
    but can run in distributed way)

    \item bug fixed: when used in distributed environment, variable
    name info is missing, therefore, "Error: Training Set is empty"
    occurs.

  }
}
