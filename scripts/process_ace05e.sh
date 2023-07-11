export DYGIEFORMAT_PATH="./processed_data/ace05e_dygieppformat"
export OUTPUT_PATH="./processed_data/ace05e_bert"
mkdir $OUTPUT_PATH

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/train.json -o $OUTPUT_PATH/train.json -b bert-large-cased -w 1

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/dev.json -o $OUTPUT_PATH/dev.json -b bert-large-cased -w 1

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/test.json -o $OUTPUT_PATH/test.json -b bert-large-cased -w 1