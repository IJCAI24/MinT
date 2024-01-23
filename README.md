# Medical Match, Personal Touch: A Dual-Module Approach for Tailored Doctor Recommendations
This is the implementation for our IJCAI 2024 paper:
>Medical Match, Personal Touch: A Dual-Module Approach for Tailored Doctor Recommendations.

## Environment
Please follow `requirements.txt`

## Run *MinT*
Process dataset:
```
python ./Script/dataset_process.py --dataset=lung
```
Training:
```
python ./Mint/main.py --dataset=lung --alpha=0.8 --beta=0.5 --lr=0.001
```
