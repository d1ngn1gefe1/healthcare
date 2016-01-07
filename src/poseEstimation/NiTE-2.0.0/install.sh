#!/bin/sh

#/****************************************************************************
#*                                                                           *
#*  NITE 2.x Alpha                                                           *
#*  Copyright (C) 2012 PrimeSense Ltd.                                       *
#*                                                                           *
#*  This file is part of NiTE2.                                              *
#*                                                                           *
#****************************************************************************/

ORIG_PATH=`pwd`
cd `dirname $0`
SCRIPT_PATH=`pwd`
cd $ORIG_PATH

OUT_FILE="$SCRIPT_PATH/NiTEDevEnvironment"

echo "export NITE2_INCLUDE=$SCRIPT_PATH/Include" > $OUT_FILE
if [ `uname -m` = "x86_64" ];
then
        echo "export NITE2_REDIST64=$SCRIPT_PATH/Redist" >> $OUT_FILE
else
        echo "export NITE2_REDIST=$SCRIPT_PATH/Redist" >> $OUT_FILE
fi
chmod a+r $OUT_FILE
