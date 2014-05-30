\name{NEWS}
\title{News for Package 'wsrf'}

\newcommand{\cpkg}{\href{http://CRAN.R-project.org/package=#1}{\pkg{#1}}}

\section{Changes in wsrf version 1.4.0 (2014-05-28)}{


  \subsection{NEW FEATURES}{
    \itemize{

      \item Add new function \code{wsrfParallelInfo} for querying the
      underlying parallel implementation (parallel or not)
      
    }
  }

  \subsection{BUG FIXES}{
    \itemize{
      
      \item The predict method requires a dummy outcome column to be in the
      new data (Thank Max Knhn for reporting this bug)
      
    }
  }

  \subsection{MINOR CHANGES}{
    \itemize{

      \item Adapt R's make variable \env{CXX\_STD} to configure
      installation
      
      \item List packages used by vignettes code in DESCRIPTION
    }
  }

  \subsection{DOCUMENTATION}{
    \itemize{

      \item Let R to build the vignettes so that it can be visible on
      \cpkg{wsrf}
      
    }
  }

}


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
