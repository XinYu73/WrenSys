<!--
 * @Author: Yu Xin
 * @Date: 2025-02-28 14:08:24
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2025-02-28 16:03:53
 * @Description: 
-->


1. run remove_duplicate.py on 
   1. experimental + unique_experimental
   2. theoretical + unique_theoretical
2. mv unique_experimental/unique_experimental.csv .
3. mv unique_theoretical/unique_theoretical.csv .
4. run remove_synthesized_from_theoretical.py
5. run split_data.py