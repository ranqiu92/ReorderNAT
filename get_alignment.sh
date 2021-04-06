#!/bin/bash

FASTALIGN=$1

TRAIN_SRC=$2
TRAIN_TGT=$3
DISTILLED_TGT=$4

OUTPUT_DIR=$5
TMP_DIR=${OUTPUT_DIR}/tmp

mkdir -p ${TMP_DIR}

paste ${TRAIN_SRC} ${TRAIN_TGT} | sed "s/$(printf '\t')/ ||| /g" > ${TMP_DIR}/train.concat
paste ${TRAIN_SRC} ${DISTILLED_TGT} | sed "s/$(printf '\t')/ ||| /g" > ${TMP_DIR}/distilled.concat

cat ${TMP_DIR}/train.concat  \
    ${TMP_DIR}/distilled.concat > ${TMP_DIR}/concat.full

$FASTALIGN \
    -i ${TMP_DIR}/concat.full \
    -v -p ${TMP_DIR}/concat.full.cond \
    -d -o \
    > ${TMP_DIR}/concat.full.aligned


sample_num=$(wc -l < ${TRAIN_SRC})

tail -n${sample_num} ${TMP_DIR}/concat.full.aligned > ${OUTPUT_DIR}/distilled.aligned

rm -r ${TMP_DIR}
