# Monthly
python RealityMining.py -t RealityMining_1_month_undir -t_int month -l 11 -lb 2 -bs 100 -iter 35 -sm 5000 -exp lp -method dynAERNN &

# Weekly
python RealityMining.py -t RealityMining_1_week_agg_undir -t_int week -l 50 -lb 2 -bs 100 -iter 35 -sm 5000 -exp lp -method dynAERNN &

