# CONFIGURE BUILD SYSTEM
TARGET     = cpd$(ALTO_MASK_LENGTH)
BUILD_DIR  = ./build-$(ALTO_MASK_LENGTH)
INC_DIR    = ./include
SRC_DIR    = ./src
TEST_DIR   = ./tests
MAKE_DIR   = ./mk
Q         ?= @


##########################################
# DO NOT EDIT BELOW
include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk

space := $(eval) $(eval)
ifneq ($(strip $(MODES_SPECIALIZED)),0)
  MODES_SPECIALIZED := 0,$(MODES_SPECIALIZED)
endif
ifneq ($(strip $(RANKS_SPECIALIZED)),0)
  RANKS_SPECIALIZED := 0,$(RANKS_SPECIALIZED)
endif
DEFINES += -DALTO_MODES_SPECIALIZED=$(subst $(space),,$(MODES_SPECIALIZED))
DEFINES += -DALTO_RANKS_SPECIALIZED=$(subst $(space),,$(RANKS_SPECIALIZED))
ifeq ($(THP_PRE_ALLOCATION),true)
DEFINES += -DALTO_PRE_ALLOC
endif
ifeq ($(ALTO_DYNAMIC_THREADS),true)
DEFINES += -DALTO_DTHREADS
endif
ifeq ($(ALTERNATIVE_PEXT),true)
DEFINES += -DALT_PEXT
endif
ifeq ($(MEMTRACE),true)
DEFINES += -DALTO_MEM_TRACE
endif
ifeq ($(DEBUG),true)
DEFINES += -DALTO_DEBUG
endif
ifeq ($(BLAS_LIBRARY),MKL)
DEFINES += -DMKL
endif

# For testing...
DEFINES += -DALTO_TEST_DATASETS=$(TEST_DIR)/tensors/

INCLUDES += -I$(INC_DIR)
TEST_INCLUDE := $(INCLUDES) -I./$(TEST_DIR)/include

ENTRYPOINT	:= main
TEST_TARGET	:= test

SRC        = $(wildcard $(SRC_DIR)/*.cpp)
ASM        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.s,$(SRC))
OBJ        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o,$(SRC))

# Added for test
TEST_SOURCES	:= $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS	:= $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)

nil :=
OBJ_AND_TEST_OBJS = $(nil)
# All from src/
OBJ_AND_TEST_OBJS += $(filter-out \ $(BUILD_DIR)/$(ENTRYPOINT).o, $(OBJ))
# Additional from tests/
OBJ_AND_TEST_OBJS += $(TEST_OBJECTS)

CPPFLAGS  := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)

define speaker
	@echo [make:$$PPID] $(1)
	@$(1)
endef

$(TARGET): $(BUILD_DIR) $(OBJ)
	@echo "===>  LINKING  $(TARGET)"
	$(Q)$(CXX) $(LFLAGS) -o $(TARGET) $(OBJ) $(LIBS)

asm: $(BUILD_DIR) $(ASM)

info:
	@echo $(CXXFLAGS)
	$(Q)$(CXX) $(VERSION)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | build_dir
	$(Q)$(CXX) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $@

$(BUILD_DIR)/%.o:  $(SRC_DIR)/%.cpp
	@echo "===>  COMPILE  $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/%.s:  $(SRC_DIR)/%.c
	@echo "===>  GENERATE ASM  $@"
	$(CXX) -S $(CPPFLAGS) $(CXXFLAGS) $< -o $@

.PHONY: build_dir
build_dir: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@echo "===> TEST::COMPILE $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(TEST_INCLUDE)

$(TEST_TARGET): $(TARGET) $(OBJ_AND_TEST_OBJS)
	@echo "==> CREATING TEST $(TEST_TARGET)"
	$(call speaker,\
	$(Q)$(CXX) $(LFLAGS) -o $(TEST_TARGET) $(OBJ_AND_TEST_OBJS) $(LIBS))

debug: $(OBJ_AND_TEST_OBJS)
	@echo $@
	#@echo ${TEST_OBJECTS}
	#@echo ${TEST_SOURCES}
	#@echo $(OBJ_AND_TEST_OBJS)
.PHONY: clean distclean

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
