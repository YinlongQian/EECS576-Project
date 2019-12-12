# RealtityMining is a directed network! 
# Testing different lookback values given the default values 
#1 month, lb = 2, no aggregation, dim =128
python RealityMining_embedding.py -t RealityMining_equal_dir -t_int equal -is_agg False -l 51 -lb 2 -bs 35 -iter 25 -sm 10057 -exp save_emb -method dynAERNN 


#python RealityMining_embedding.py -t RealityMining_1_week_dir -t_int week -is_agg False -l 50 -lb 2 -bs 35 -iter 25 -sm 10057 -exp save_emb -method dynAERNN 


#python RealityMining_embedding.py -t RealityMining_1_month_dir -t_int month -is_agg False -l 11 -lb 4 -bs 35 -iter 25 -sm 10057 -exp save_emb -method dynAERNN 

#python RealityMining_embedding.py -t RealityMining_1_week_dir -t_int week -is_agg False -l 50 -lb 4 -bs 35 -iter 25 -sm 10057 -exp save_emb -method dynAERNN 


#1 month, lb = 4, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_month_dir -t_int month -is_agg False -l 11 -lb 4 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 


#1 month, lb = 6, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_month_dir -t_int month -is_agg False -l 11 -lb 6 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 


#1 month, lb = 8, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_month_dir -t_int month -is_agg False -l 11 -lb 8 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 

# Weekly

#1 week, lb = 2, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_week_dir -t_int week -l 50 -is_agg False -lb 2 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 

#1 week, lb = 4, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_week_dir -t_int week -l 50 -is_agg False -lb 4 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 

#1 week, lb = 6, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_week_dir -t_int week -l 50 -is_agg False -lb 6 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 


#1 week, lb = 8, no aggregation, dim =128
#python RealityMining_embedding.py -t RealityMining_1_week_undir -t_int week -l 50 -is_agg False -lb 8 -bs 35 -iter 25 -sm 2000 -exp save_emb -method dynAERNN 
