include ../BuildSystem/CommonDefs.mak

BIN_DIR = ../Bin

INC_DIRS = \
	../../Include \
	../GL \
	$(OPENNI2_INCLUDE)

SRC_FILES = *.cpp

ifeq ("$(OSTYPE)","Darwin")
	CFLAGS += -DMACOS
	LDFLAGS += -framework OpenGL -framework GLUT
else
	CFLAGS += -DUNIX -DGLX_GLXEXT_LEGACY
	USED_LIBS += glut GL
endif

LIB_DIRS += $(OPENNI2_REDIST)

USED_LIBS += OpenNI2 NiTE2

EXE_NAME = HandViewer

CFLAGS += -Wall

include ../BuildSystem/CommonCppMakefile
