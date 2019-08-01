![Alt text](https://www.deepblueai.com/usr/deepblue/v3/images/logo.png "DeepBlue")  
[![license](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://github.com/DeepBlueAI/AutoSmart/blob/master/LICENSE)
# The introduction of AutoSmart
The 1st place solution for KDD Cup 2019 AutoML Track

# How to install
```python
pip install AutoSmart
```
# How to use
```python
import auto_smart

info = auto_smart.read_info("data")
train_data,train_label = auto_smart.read_train("data",info)
test_data = auto_smart.read_test("data",info)
auto_smart.train_and_predict(train_data,train_label,info,test_data)
```
# Data Sample
### Data

This page describes the datasets that our system can deal with.

#### Components
Each dataset is split into two subsets, namely the training set and the testing set.

Both sets have:

- a **main table file** that stores the **main table** (label excluded);
- multiple **related table files** that store the **related tables**;
- an **info dictionary** that contains important information about the dataset, including table relations;
- The training set has an additional **label file** that stores **labels** associated with the **main table**.

### Table files

Each **table file** is a CSV file that stores a table (main or related), with '**\t**' as the delimiter. The first row indicates the names of features, a.k.a 'schema', and the following rows are the records.

The type of each feature can be found in the info dictionary that will be introduced soon. 

There are 4 types of features, indicated by "cat", "num", "multi-cat", and "time", respectively:

- **cat**: categorical feature, an integer
- **num**: numerical Feature, a real value.
- **multi-cat**: multi-value categorical Feature: a set of integers, split by the comma. The size of the set is not fixed and can be different for each instance. For example, topics of an article, words in a title, items bought by a user and so on.
- **time**: time feature, an integer that indicates the timestamp.


### Label file
The **label file** is associated only with the main table in the training set. It is a CSV file that contains only one column, with the first row as the header and the remaining indicating labels associated with instances in the main table.

### info dictionary
Important information about each dataset is stored in a python dictionary structure named as **info**, which acts as an input of this system. Generally,you need to manually generate the dictionary information info.json file. Here we give details about info.

![Alt text](https://i.ibb.co/4dQxCRD/info.png "datainfo")  

Descriptions of the keys in info:

- **time_budget**: time budget for this dataset (sec). 
- **time_col**: the column name of the primary timestamp; Each dataset has one unique time_col; time_col is definitely contained in the main table, but not necessarily in a related table;
- **start_time**: DEPRECATED.
- **tables**: a dictionary that stores information about tables. Each key indicates a table, and its corresponding value is a dictionary that indicates the type of each column in this table. Two kinds of keys are contained in tables:
    - **main**: the main table;
    - **table_{i}**: the i-th related table.
    - There are 4 types of features, indicated by "cat", "num", "multi-cat", and "time", respectively:
        - **cat**: categorical feature, an integer
        - **num**: numerical Feature, a real value.
        - **multi-cat**: multi-value categorical Feature: a set of integers, split by the comma. The size of the set is not fixed and can be different for each instance. For example, topics of an article, words in a title, items bought by a user and so on.
        - **time**: time feature, an integer that indicates the timestamp.

- **relations**: a list that stores table relations in the dataset. Each relation can be represented as an ordered table pair (**table_A**, **table_B**), a key column **key** that appears in both tables and acts as the pivot of table joining, and a relation type **type**. Different relation types will be introduced shortly.

#### Relations Between Tables
Four table relations are considered in this system:

- **one-to-one** (1-1): the key columns in both **table_A** and **table_B** have no duplicated values;
- **one-to-many** (1-M): the key column in **table_A** has no duplicated values, but that in **table_B** may have duplicated values;
- **many-to-one** (M-1): the key column in **table_A** may have duplicated values, but that in **table_B** has no duplicated values;
- **many-to-many** (M-M): the key columns in both **table_A** and **table_B** may have duplicated values.




# Contact Us
DeepBlueAI: 1229991666@qq.com
