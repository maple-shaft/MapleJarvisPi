#!/bin/bash

# Base models need to be copied to the venv directory so that openwakeword can function.

basepath=./.venv/lib/python3.10/site-packages/openwakeword
mkdir $basepath/resources
mkdir $basepath/resources/models
cp models/* $basepath/resources/models