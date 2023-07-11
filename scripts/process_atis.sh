## Base Setting

export RAW_DATA_PATH="./atis"

export OUTPUT_PATH="./processed_data/atis"
mkdir $OUTPUT_PATH


for SPLIT in train dev test
do
    python preprocessing/process_tosp.py -i $RAW_DATA_PATH -o $OUTPUT_PATH/$SPLIT -s $SPLIT
done

