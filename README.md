# Table Understanding
### Setting up the environment
Set up the environment by referring to [ISI's Table Understanding](http://github.com/usc-isi-i2/table-understanding/tree/impl/)

Once the environment is set up, check for conflicting package versions by using the follwing conda command:
    ```conda install --dry-run --file environment.yaml```

For the environment to run without conflicting packages, run conda install for the following packages with their respective versions:
- numpy==1.19.0
- spacy==3.3.1
- scikit-learn==0.23.2
- blis==0.7.11
- pydantic==1.7.4
- pslpython==2.3.2
- thinc==8.0.7


### PSL module, ce_model:
- Download the [PSL Module](https://drive.google.com/file/d/1ndVTP3WSG8OLoDjYnePvuVZ5fxXBCyRz/view?usp=sharing), and [ce_models](https://drive.google.com/uc?id=1DJfEgqoHzfQYBllzey21zS39ui_kwId-) 
- Place the PSL module, ce_model, and input data files in the appropriate directories as shown in the following snapshot (note: /tmp/ will be generated during runtime):

### Inputs:
The inputs are generated using Jupyter notebooks and stored at:
/isi-table-understanding/data/tu_pipeline_test

### Running table-understanding pipelines
To run a pipeline, use the following command: ```make run PIPELINE=<pipeline name>```
Pipelines are defined in the ```./pipelines``` package and use ```./tmp``` as a working directory to download models/data or store outputs.

<!-- ![alt text](https://github.com/yugal82/isi-table-understanding/blob/main/data%20File%20Arrangement.png) -->
<!-- ![alt text](https://github.com/doswal/isi-table-understanding/blob/main/tmp%20File%20Arrangement.png) -->

### Troubleshoot 1: Infersent module:
- For troubleshooting the infersent module, import from [here](https://github.com/facebookresearch/InferSent/blob/main/models.py) in the tabular_cell_type_classification>src>helpers.py file

### Output:
The pipeline takes Excel file paths and produces a colorized version of the same file, along with a corresponding ```.yaml``` file. This output contains data about row (start, end) and column (start, end) positions. Outputs are usually stored in the temporary "tmp" folder, though the current outputs can be found in the ```root > output``` folder.

### Data block printing:
For extracting JSON/txt files or printing data series per block, refer to the Jupyter Notebook attached.