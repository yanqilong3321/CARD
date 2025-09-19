## Reproduce the results

Our experiment settings are specified in the `CARD.yml` file. To create the environment, you can use the following command:

```
conda env create -f card.yml
```

### Zhihu
```
python -u CARD.py --data zhihu --strategy pred_future --stability_threshold 1.8893 --future_window_size 3 --continuity_threshold 0.1 --score_temp 1.0 --timesteps 1000 --lr 0.01 --beta_sche linear --w 6 --cuda 0 --eval 5 --optimizer adamw --diffuser_type mlp1 --random_seed 100 >> log/CARD_zhihu.log 2>&1 &
```

### KuaiRec

```
python -u CARD.py --data ks    --strategy pred_future --stability_threshold 1.8893 --future_window_size 3 --continuity_threshold 0.1 --score_temp 1.0 --epoch 30 --timesteps 2000 --lr 0.00005 --beta_sche linear --w 2 --cuda 1 --eval 5 --optimizer adamw --diffuser_type mlp1 --random_seed 100 --linespace 100 >> log/CARD_ks.log 2>&1 &
```
