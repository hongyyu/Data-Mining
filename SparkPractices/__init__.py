# Not allow to use pandas for task2
# import pandas as pd
# # Import json file as dataframe
# no_spark_time = timer()        # time start without spark
# review_df = pd.read_json(review_file, lines=True)
# business_df = pd.read_json(business_file, lines=True)
#
# # review_df only need id and stars, and extract categories from business_df
# review_id_stars = review_df[['business_id', 'stars']].copy()
# business_id_category = business_df[['business_id', 'categories']].copy()
#
# # Define function for check none in categories and split string into list without trailing spaces
# def check_none_and_split(line):
#     if line is not None:
#         return [cate.strip() for cate in line.split(',')]
#
# # Separate categories with corresponding business_id
# category_df = business_id_category.categories \
#     .apply(check_none_and_split) \
#     .apply(pd.Series) \
#     .merge(business_id_category, right_index=True, left_index=True) \
#     .drop(['categories'], axis=1) \
#     .melt(id_vars = 'business_id', value_name = 'category') \
#     .drop(['variable'], axis=1) \
#     .dropna()
#
# # Join category_df and review_id_stars on business_id and then count avg sort in descending order
# join_df = pd.merge(review_id_stars,
#                    category_df,
#                    on='business_id') \
#     .groupby('category') \
#     .mean() \
#     .sort_values(['stars','category'], ascending=[False, True])
#
# print('Top %d categories with the highest average stars:' % n)
# print(join_df[:n])
#
# # Count time
# timer(no_spark_time)
#
# # Setting res
# c_name, star_avg = join_df.reset_index()['category'].to_list(), join_df['stars'].to_list()
# for i in range(n):
#     res.append([c_name[i], star_avg[i]])