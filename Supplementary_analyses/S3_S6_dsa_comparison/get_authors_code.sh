#!/bin/zsh
# Download the published code of KhudaBukhsh et al. (SDS.Epidemic, R) and Di Lauro et al. (DSA_refactor, Python) that S4–S6
# compare against, at the commits we ran, and install it into external/. Their code is not distributed with this repository.
#   external/SDS.Epidemic, external/Rlib    github.com/cbskust/SDS.Epidemic @ 3ecb29d; its SDSMCMC_0.1.0.tar.gz is installed
#                                           into Rlib with the current CRAN versions of its dependencies. The versions we used
#                                           are in sds_r_versions.txt; the script lists any installed version that differs.
#   external/DSA_refactor, external/.venv   github.com/Zkeggia/DSA_refactor @ 26671d6; Python 3.14 environment with the package
#                                           versions we used (dsa_refactor_requirements.txt)
# external/data (datasets written by s6_0_export.jl) and external/shim/pyDOE.py (our one-line module alias: their code imports
# pyDOE, now published as pydoe) are ours. run_method_supplement.sh S6 runs this script when the code is missing.
# Usage: zsh S3_S6_dsa_comparison/get_authors_code.sh   (EXT=<folder> installs elsewhere; PYTHON=<interpreter>, default python3.14)
set -e
cd "$(dirname "$0")"
X=${EXT:-external}; mkdir -p $X/Rlib
fetch() {   # fetch <repository> <folder> <commit>
  [[ -d $X/$2/.git ]] || git clone -q $1 $X/$2
  git -C $X/$2 -c advice.detachedHead=false checkout -q $3
  echo "$2 @ $(git -C $X/$2 rev-parse --short HEAD)"
}
fetch https://github.com/cbskust/SDS.Epidemic SDS.Epidemic 3ecb29d85219a03f70567123e98c7e4632038508
fetch https://github.com/Zkeggia/DSA_refactor DSA_refactor 26671d6d7f5d7d21af45682e0685a51cd596848f

Rscript -e '
  a <- commandArgs(TRUE); lib <- a[1]; .libPaths(c(lib, .libPaths()))
  need <- setdiff(c("plyr", "smfsb", "ramcmc"), rownames(installed.packages(lib.loc = lib)))
  if (length(need)) install.packages(need, lib = lib, repos = "https://cloud.r-project.org", quiet = TRUE)
  install.packages(a[2], lib = lib, repos = NULL, type = "source", quiet = TRUE)
  suppressMessages(library(SDSMCMC))
  used <- read.table(a[3], comment.char = "#", col.names = c("pkg", "ver"), colClasses = "character")
  now <- sapply(used$pkg, function(p) if (p == "R") paste(R.version$major, R.version$minor, sep = ".")
                else tryCatch(as.character(packageVersion(p)), error = function(e) "not installed"))
  same <- mapply(function(u, n) n != "not installed" && package_version(u) == package_version(n), used$ver, now)
  cat(sprintf("%-14s used %-10s installed %s%s\n", used$pkg, used$ver, now, ifelse(same, "", "   <- differs")), sep = "")
  if (!all(same)) cat("\nSome versions differ from those used in the supplement (", basename(a[3]), "). To install the package version we used: ",
                      "remotes::install_version(\"<package>\", \"<version>\", lib = \"", lib, "\")\n", sep = "")
' $X/Rlib $X/SDS.Epidemic/SDSMCMC_0.1.0.tar.gz sds_r_versions.txt

[[ -x $X/.venv/bin/python ]] || ${PYTHON:-python3.14} -m venv $X/.venv
$X/.venv/bin/python -m pip install -q -r dsa_refactor_requirements.txt
PYTHONPATH=external/shim:$X/DSA_refactor $X/.venv/bin/python -W ignore -c \
  "from Likelihood import log_likelihood_models; import sys, numpy, scipy, pyswarms; print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| pyswarms', pyswarms.__version__)"
