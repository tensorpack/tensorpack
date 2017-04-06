# $File: Makefile
# $Date: Wed Jun 17 20:52:38 2015 +0800

OBJ_DIR = obj

CXX ?= g++

OPTFLAGS ?= -O3 -march=native
#OPTFLAGS ?= -g3 -fsanitize=address,undefined -O2 -lasan
#OPTFLAGS ?= -g3 -fsanitize=leak -O2 -lubsan

# optional extra packages
#LIBS = opencv
#INCLUDE_DIR += $(shell pkg-config --cflags $(LIBS))
#LDFLAGS += $(shell pkg-config $(LIBS) --libs)

CXXFLAGS ?=
CXXFLAGS += $(INCLUDE_DIR)
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers
CXXFLAGS += $(DEFINES) -std=c++11 $(OPTFLAGS) -fPIC
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 		# https://github.com/tensorflow/tensorflow/issues/1569

LDFLAGS += $(OPTFLAGS)
LDFLAGS += -lzmq -lprotobuf $(OMP_FLAG)
LDFLAGS += -shared -fPIC

SHELL = bash
# sources to include
ccSOURCES = $(shell find $(SRCDIRS) -name "*.cc" | sed 's/^\.\///g')
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

SO = $(ccSOURCES:.cc=.so)

.PHONY: all clean

all: $(SO)

ifneq ($(MAKECMDGOALS), clean)
sinclude $(DEPFILES)
endif

%.so: $(OBJ_DIR)/%.o
	@echo "Linking $@ ..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "done."

$(OBJ_DIR)/%.o: %.cc
	@echo "[cc] $< ..."
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

$(OBJ_DIR)/%.d: %.cc Makefile
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<"  > "$@" || rm "$@"

clean:
	@rm -rvf $(OBJ_DIR) $(SO)

