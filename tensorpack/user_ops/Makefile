# $File: Makefile
# $Date: Tue Oct 31 11:44:27 2017 +0800

OBJ_DIR = obj

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
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(DEFINES) -std=c++11 $(OPTFLAGS) -fPIC

# TODO https://github.com/tensorflow/tensorflow/issues/1569
# You may need to disable this flag if you compile tensorflow yourself with gcc>=5
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0

LDFLAGS += $(OPTFLAGS)
LDFLAGS += -shared -fPIC
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -Wl,-undefined -Wl,dynamic_lookup
endif

SHELL = bash
# sources to include
ccSOURCES = $(shell find $(SRCDIRS) -name "*.cc" | sed 's/^\.\///g')
OBJS = $(addprefix $(OBJ_DIR)/,$(ccSOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

# TODO what about mac?
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

