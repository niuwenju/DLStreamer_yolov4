#!/bin/bash
# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e

PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../../python \
python3 $(dirname "$0")/yolov4.py -i test_video/sample_1080p_h264.mp4 -d model/yolov4/FP16/yolov4.xml
