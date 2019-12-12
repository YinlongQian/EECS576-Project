# 2 year, no agg, undir 
python dblp_embedding.py -t dblp_1_year_undir -t_int 1 -is_agg False -is_dir False -l 78 -lb 2 -bs 100 -iter 35 -sm 7500 -exp lp -method dynAERNN &

# 1 year, agg, undir
python dblp_embedding.py -t dblp_1_year_agg_undir -t_int 1 -is_agg True -is_dir False -l 78 -lb 2 -bs 100 -iter 35 -sm 7500 -exp lp -method dynAERNN &

# 5 year, no agg, undir
python dblp_embedding.py -t dblp_5_year_undir -t_int 5 -is_agg False -is_dir False -l 16 -lb 2 -bs 100 -iter 30 -sm 7500 -exp lp -method dynAERNN &

# 5 year, agg, undir 
python dblp_embedding.py -t dblp_5_year_agg_undir -t_int 5 -is_agg True -is_dir False -l 16 -lb 2 -bs 100 -iter 30 -sm 7500 -exp lp -method dynAERNN &
