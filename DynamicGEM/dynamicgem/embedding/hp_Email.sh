# Email is a directed network! 
# Testing different lookback values given the default values 
#1 month, lb = 2, no aggregation, dim =128
#python Email_embedding.py -t Email_1_month_dir -t_int month -is_agg False -l 17 -lb 2 -bs 25 -iter 25 -sm 1005 -exp save_emb -method dynAERNN 

#python Email_embedding.py -t Email_1_week_dir -t_int week -is_agg False -l 74 -lb 2 -bs 25 -iter 25 -sm 1005 -exp save_emb -method dynAERNN 


#python Email_embedding.py -t Email_equal_month_dir -t_int equal_monthly -is_agg False -l 17 -lb 2 -bs 25 -iter 25 -sm 1005 -exp save_emb -method dynAERNN 

python Email_embedding.py -t Email_equal_week_dir -t_int equal_weekly -is_agg False -l 74 -lb 2 -bs 25 -iter 25 -sm 1005 -exp save_emb -method dynAERNN 


