# Dacon Tour2022
Train and test code of team hyy (private learderboard 9th) for 2022 관광데이터 AI 경진대회

https://dacon.io/competitions/official/235978/overview/description

Only text information is used

Method description please refer to `tour2022(team_hyy)(2022.11.02).pdf`

## Requirements

- pandas
- numpy
- scikit-learn
- pytorch >=1.12.1
- transformers >=4.22.2

## Usage

Run following script by changing line 319~324 to your own data path.

You should run the script 5 times by setting seed from 40 to 44. Because we will do ensemble on the 5 results.

GPU with larger than 24GB memory is required. However, if your memory is insufficient, you can reduce batch_size or max_length to fit to your device.

It will take nearly 20 hours for one run.
```
python train_and_infer.py
```

After run 5 times, you will get 5 result folders (model_seed_40, model_seed_41...)

And then run following script by changing line 7~21 to your own data path to get ensemble of the 5 results.
```
python ensemble_seed_wise_results.py
```
The output result file `submit.csv` in the `seed_wise_ensemble` folder yields public leaderboard score: 0.86543, private leaderboard score: 0.85923.

We also have tried knowledge distillation learning using the predicted ensembled test result. 

We mixed the test data and train data to train a new model where the label for the test data is the ensembled predicted probability, in the `seed_wise_ensemble` folder, of test sample.

Run following script to do knowledge distillation. Change line 359~365 to your own data path.

```
python train_and_infer_distill.py
```
The output result file `distill_submit_5fold.csv` yields public leaderboard score: 0.86447, private leaderboard score: 0.86012.




