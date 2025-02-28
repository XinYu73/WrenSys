<!--
 * @Author: Yu Xin
 * @Date: 2025-02-28 14:08:24
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2025-02-28 16:43:34
 * @Description: 
-->


1. run remove_duplicate.py on 
   1. experimental + unique_experimental
   2. theoretical + unique_theoretical
2. mv unique_experimental/unique_experimental.csv .
3. mv unique_theoretical/unique_theoretical.csv .
4. run remove_synthesized_from_theoretical.py
5. run split_data.py
6. drop structures
   ```python
   import pandas as pd
   import os
   data = pd.read_csv('train_set.csv',header=None)
   data.loc[data[0].apply(lambda x: os.path.exists(f'cif_files/{x}.cif'))].to_csv(f'cif_files/id_prop.csv',header=None,index=None)
   ```