# $File: Makefile
# $Date: Tue Dec 12 18:04:22 2017 -0800

OBJ_DIR = obj
PYTHON = python

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX ?= g++
endif
ifeq ($(UNAME_S),Darwin)
	CXX ?= clang++
endif

OPTFLAGS ?= -O3 -march=native
#OPTFLAGS ?= -g3 -fsanitize=address,undefined -O2 -lasan
#OPTFLAGS ?= -g3 -fsanitize=leak -O2 -lubsan

# extra packages from pkg-config
LIBS = libzmq
INCLUDE_DIR += $(shell pkg-config --cflags $(LIBS))
LDFLAGS += $(shell pkg-config $(LIBS) --libs)

CXXFLAGS += $(INCLUDE_DIR)
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CXXFLAGS += $(DEFINES) -std=c++11 $(OPTFLAGS) -fPIC

ifneq ($(MAKECMDGOALS), clean)
TF_CXXFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
endif

CXXFLAGS += $(TF_CXXFLAGS)

LDFLAGS += $(OPTFLAGS)
LDFLAGS += -shared -fPIC
LDFLAGS += $(TF_LDFLAGS)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -Wl,-undefined -Wl,dynamic_lookup
endif

SHELL = bash
# sources to include
ccSOURCES = $(shell find $(SRCDIRS) -name "*.cc" | sed 's/^\.\///g')
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

# TODO what about mac?
SO = zmq_recv_op.so

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

