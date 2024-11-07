import numpy as np
import pandas as pd

product_data = {"Product_ID":[1,2,3,4],
               "Product_Name":["Home entertainment","Computer","Phone","Security"],
               "Category":["Electronics","Electronics","Electronics","Electronics"],
               "Brand":["BrandA","BrandB","BrandC","BrandD"] }

product_df = pd.DataFrame(product_data)

time_data = {"Time_ID":[1,2,3,4],
            "Date":["01-01-2024","14-01-2024","22-02-2024","16-03-2024"],
            "Month":["January","January","February","March"],
            "Quarter":["Q1","Q1","Q1","Q1"],
            "Year":["2024","2024","2024","2024"]}

time_df = pd.DataFrame(time_data)

store_data = {"Store_ID":[1,2,3,4],
             "Store_Location":["Vancouver","Toronto","NewYork","Chicago"],
             "Store_Name":["StoreA","StoreA","StoreB","StoreB"],
             "Country":["Canada","Canada","USA","USA"]}

store_df = pd.DataFrame(store_data)

sales_data = {"Sales_ID":[1,2,3,4],
             "Product_ID":[1,2,3,4],
             "Time_ID":[1,2,3,4],
             "Store_ID":[1,2,3,4],
             "Sales":[10,20,30,40],
             "Revenue":[1000,2000,3000,4000]}

sales_df = pd.DataFrame(sales_data)

print("Product Dimension:\n",product_df)
print("Time Dimension:\n",time_df)
print("Store Dimension:\n",store_df)
print("Sales Dimension:\n",sales_df)


#roll up
rollup_df = sales_df.merge(store_df,on="Store_ID").groupby(["Country","Store_Name"]).agg({"Sales":'sum',"Revenue":'sum'})
print("Roll up results:\n",rollup_df)

#slice
slice_df = sales_df.merge(product_df,on="Product_ID").query("Product_Name == 'Home entertainment'")
print("slice operation result:\n",slice_df)


#dice
dice_df = sales_df.merge(time_df,on= "Time_ID").merge(store_df,on="Store_ID").query("Quarter == 'Q1' and Country in ['Canada'] and Store_Name  == 'StoreA'")
print("dice operation result:\n",dice_df)
