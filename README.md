# multi_view_radiology

## File Structure
```
.
├── config_template.py
├── data_loader.py
├── dicom_image.py
├── main.py
├── model.py
├── preprocessing
│   └── Data_Processing_Open_i.py
├── README.md
├── test
│   └── test_data_loader.py
├── vocab.pkl
└── vocabulary.py
```
## Preprocessing for Open-i dataset
**Data_Preproceesing_Open_i.py** has to be placed within Open_i dataset folder to generate .csv file

## Config file
**config_template.py** should be renamed **config.py** and updated with system-specific parameters like file paths.

## Execution
```bash
$python Main.py
```

## Reading DICOM files
Requires `pydicom`, available on conda-forge and PyPI
`d = pydicom.dcmread(file_path)` returns `pydicom.dataset.FileDataset` object:
https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.dataset.FileDataset.html

Use `d.PatientOrientation` or `d.ViewPosition` for orientation metadata.
