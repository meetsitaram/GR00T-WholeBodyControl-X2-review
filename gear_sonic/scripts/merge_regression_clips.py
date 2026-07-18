#!/usr/bin/env python3
"""Merge kplanner-generated clips ahead of the reference regression clips."""
import sys, os, joblib
ref, *gen, out = sys.argv[1:]
merged = {}
for f in gen:                      # kplanner-driven clips first
    if os.path.exists(f):
        merged.update(joblib.load(f))
merged.update(joblib.load(ref))    # then the curated reference clips
joblib.dump(merged, out)
print(f"combined suite: {len(merged)} clips")
