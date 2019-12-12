# Have month and weekly granularity. Save each one's embedding.
# Need to consider multiple look back values. 
# CollegeMsg is DIRECTED 


#28 evenly spaced graphs to correspond to week based splits
# 1 equal (equal-space), lb = 2, no aggregation, dim = 128, 
#python CollegeMsg_embedding.py -t CollegeMsg_equal_dir -t_int equal -is_agg False -is_dir True -l 28 -lb 2 -bs 100 -iter 250 -sm 2000 -exp save_emb -method dynAERNN 

#echo "FINISHED EQUAL -- 2"
# 1 equal, lb = 4, no aggregation, dim = 128, 
#python CollegeMsg_embedding.py -t CollegeMsg_equal_dir -t_int equal -is_agg False -is_dir True -l 28 -lb 4 -bs 100 -iter 250 -sm 2000 -exp save_emb -method dynAERNN 


# 1 equal, lb = 6, no aggregation, dim = 128, 
#python CollegeMsg_embedding.py -t CollegeMsg_equal_dir -t_int equal -is_agg False -is_dir True -l 28 -lb 6 -bs 100 -iter 250 -sm 2000 -exp save_emb -method dynAERNN 


# 1 equal, lb = 8, no aggregation, dim = 128, 
#python CollegeMsg_embedding.py -t CollegeMsg_equal_dir -t_int equal -is_agg False -is_dir True -l 28 -lb 8 -bs 100 -iter 250 -sm 2000 -exp save_emb -method dynAERNN 

#echo "FINISHED EQUAL -- 8"
#MONTHLY 

# 1 month, lb = 2, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_month_dir -t_int month -is_agg False -is_dir True -l 7 -lb 2 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 month, lb = 3, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_month_dir -t_int month -is_agg False -is_dir True -l 7 -lb 3 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 month, lb = 4, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_month_dir -t_int month -is_agg False -is_dir True -l 7 -lb 4 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 month, lb = 7, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_month_dir -t_int month -is_agg False -is_dir True -l 7 -lb 5 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

echo "FINISHED MONTH -- 8"

#WEEKLY 

# 1 week, lb = 2, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_week_dir -t_int week -is_agg False -is_dir True -l 28 -lb 2 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 week, lb = 4, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_week_dir -t_int week -is_agg False -is_dir True -l 28 -lb 4 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 week, lb = 6, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_week_dir -t_int week -is_agg False -is_dir True -l 28 -lb 6 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

# 1 week, lb = 8, no aggregation, dim = 128, 
python CollegeMsg_embedding.py -t CollegeMsg_1_week_dir -t_int week -is_agg False -is_dir True -l 28 -lb 8 -bs 100 -iter 250 -sm 1000 -exp save_emb -method dynAERNN 

echo "FINISHED ALL"
