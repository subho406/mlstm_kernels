#!/bin/bash

# Execute the first Python script
COMMON_ARGS="--folder_suffix rerun_v0"

python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mamba $COMMON_ARGS
python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mlstm_triton $COMMON_ARGS
python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark fla $COMMON_ARGS

python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mlstm_triton $COMMON_ARGS --num_heads 16
python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark fla $COMMON_ARGS --num_heads 16

python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mlstm_triton $COMMON_ARGS --num_heads 32
python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark fla $COMMON_ARGS --num_heads 32

python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mlstm_triton $COMMON_ARGS --num_heads 64
python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark fla $COMMON_ARGS --num_heads 64

# # Check if the first script ran successfully
# if [ $? -ne 0 ]; then
#     echo "The first script encountered an error. Exiting."
#     exit 1
# fi


# # Check if the second script ran successfully
# if [ $? -ne 0 ]; then
#     echo "The second script encountered an error. Exiting."
#     exit 1
# fi

# echo "Both scripts executed successfully."
