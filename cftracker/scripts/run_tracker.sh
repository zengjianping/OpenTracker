#!/bin/bash

binaries/run_tracker.bin \
    --video_path=datas/sequences/Crossing \
    --tracker_name=ECO \
    --tracker_config=configs/tracker_config_eco.yaml \
    --select_roi=false
