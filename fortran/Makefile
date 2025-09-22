# Usage: make && make install
PYTHON ?= python
F2PY   := $(PYTHON) -m numpy.f2py
#FC     ?= x86_64-conda-linux-gnu-gfortran
FFLAGS ?= -O3 -fopenmp -J./

#CONDA_LIB ?= $(shell python -c "import sys,os;print(os.environ.get('CONDA_PREFIX','')+'/lib')")
LLIBS  := -lgomp -llapack

MOD    := cletkf_wloc
SRC    := netlib.f90 SFMT.f90 common_tools.f90 common_mtx.f90 common_letkf.f90 common_da_wloc.f90

all: clean $(MOD).so

$(MOD).so: $(SRC)
	SETUPTOOLS_USE_DISTUTILS=local NPY_DISTUTILS_APPEND_FLAGS=1 \
	$(F2PY) -c -m $(MOD) $(SRC) --fcompiler=gnu95 --opt='-O3' --f90flags='$(FFLAGS)' $(LLIBS)

install: $(MOD).so
	mkdir -p ../
	mv -v $(MOD)*.so ../

clean:
	rm -f $(MOD)*.so *.o *.mod