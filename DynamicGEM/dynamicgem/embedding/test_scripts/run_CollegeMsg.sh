
# 1 month no agg, undir 
python CollegeMsg_embedding.py -t CollegeMsg_1_month_dir -t_int month -is_agg False -is_dir True -l 7 -lb 2 -bs 10 -iter 10 -sm 500 -exp save_emb -method dynAERNN 

# 1 week no agg, undir 
#python CollegeMsg_embedding.py -t CollegeMsg_1_week_dir -t_int week -is_agg
#False -is_dir True -l 27 -lb 5 -bs 10 -iter 10 -sm 1899 -exp save_emb -method dynAERNN 


# 1 day, no agg, undir
#python CollegeMsg_embedding.py -t CollegeMsg_1_day_dir -t_int day -is_agg False -is_dir True -l 189 -lb 2 -bs 10 -iter 10 -sm 500 -exp lp -method dynAERNN &

# 5 year, agg, undir 
#python CollegeMsg_embedding.py -t CollegeMsg_5_year_agg_undir -t_int 5 -is_agg True -is_dir False -l 16 -lb 2 -bs 100 -iter 30 -sm 7500 -exp lp -method dynAERNN &
