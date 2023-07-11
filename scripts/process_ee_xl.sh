#!/bin/bash

# ACE Chinese
export ACE_PATH="../../Dataset/ace_2005_td_v7/data/"
export OUTPUT_PATH="./processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='xlm'
export PRETRAINED_TOKENIZER_NAME='xlm-roberta-large'
mkdir $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME
python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME -s resource/splits/ACE05-ZH -b $PRETRAINED_TOKENIZER_NAME -w 1 -l chinese

mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/train.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/train.json
mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/dev.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/dev.json
mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/test.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/test.json

# export TOKENIZER_NAME='mbert'
# export PRETRAINED_TOKENIZER_NAME='bert-base-multilingual-cased'
# mkdir $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME
# python preprocessing/process_ace05ep.py -i $ACE_PATH -o $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME -s resource/splits/ACE05-ZH -b $PRETRAINED_TOKENIZER_NAME -w 1 -l chinese

# mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/train.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/train.json
# mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/dev.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/dev.json
# mv $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/test.w1.oneie.json $OUTPUT_PATH/ace05XL_zh_$TOKENIZER_NAME/test.json

# ERE Spanish
export PYTHONWARNINGS=ignore

export ERE_PATH="../../Dataset/ERE_ES/"
export OUTPUT_PATH="./processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='xlm'
export PRETRAINED_TOKENIZER_NAME='xlm-roberta-large'
mkdir $OUTPUT_PATH/ereXL_es_$TOKENIZER_NAME
python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH/ereXL_es_$TOKENIZER_NAME -s resource/splits/ERE-ES -b $PRETRAINED_TOKENIZER_NAME -w 1 -l spanish

# export TOKENIZER_NAME='mbert'
# export PRETRAINED_TOKENIZER_NAME='bert-base-multilingual-cased'
# mkdir $OUTPUT_PATH/ereXL_es_$TOKENIZER_NAME
# python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH/ereXL_es_$TOKENIZER_NAME -s resource/splits/ERE-ES -b $PRETRAINED_TOKENIZER_NAME -w 1 -l spanish