# multi_view_radiology

## File Structure
```
Root
|
+----Preprocess                       Saved data 
|   | Data_Processing_Open_i.py
|   
+----data                       Saved data 
|   | png
|   | ecgen-radiology
|   
+----Main.py  
|
+----Model.py
|
+----data_loader.py
|
+----vocabulary.py
+----vocab.pkl
```
## Preprocessing for Open-i dataset
**Data_Preproceesing_Open_i.py** has to be placed within Open_i dataset folder to generate .csv file

## Config file
**config_template.py** should be renamed **config.py** and updated with system-specific parameters.

## Execution
```bash
$python Model.py
```
