include ../BuildSystem/CommonDefs.mak

BIN_DIR = ../Bin

INC_DIRS = \
	../../Include \
	../GL \
	../../../OpenNI-MacOSX-x64-2.2/Include
	#$(OPENNI2_INCLUDE)

SRC_FILES = *.cpp

ifeq ("$(OSTYPE)","Darwin")
	CFLAGS += -DMACOS
	LDFLAGS += -framework OpenGL -framework GLUT
else
	CFLAGS += -DUNIX -DGLX_GLXEXT_LEGACY
	USED_LIBS += glut GL
endif

LIB_DIRS += ../../../OpenNI-MacOSX-x64-2.2/Redist #$(OPENNI2_REDIST)

USED_LIBS += OpenNI2 NiTE2

EXE_NAME = UserViewer

CFLAGS += -Wall

include ../BuildSystem/CommonCppMakefile
