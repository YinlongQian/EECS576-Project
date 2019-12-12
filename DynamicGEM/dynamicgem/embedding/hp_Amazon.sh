# Amazon_food is a directed network! 
# Testing different lookback values given the default values 
#1 month, lb = 2, no aggregation, dim =128
python Amazon_food_embedding.py -t Amazon_food_1_month_dir -t_int monthly -is_agg False -l 20 -lb 2 -bs 25 -iter 25 -sm 23394 -exp save_emb -method dynAERNN 

python Amazon_food_embedding.py -t Amazon_food_1_week_dir -t_int weekly -is_agg False -l 20 -lb 2 -bs 25 -iter 25 -sm 23394 -exp save_emb -method dynAERNN 


python Amazon_food_embedding.py -t Amazon_food_equal_month_dir -t_int equal_monthly -is_agg False -l 20 -lb 2 -bs 25 -iter 25 -sm 23394 -exp save_emb -method dynAERNN 

python Amazon_food_embedding.py -t Amazon_food_equal_week_dir -t_int equal_weekly -is_agg False -l 20 -lb 2 -bs 25 -iter 25 -sm 23394 -exp save_emb -method dynAERNN 


