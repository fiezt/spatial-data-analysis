# Publications
This work is associated with the following publications:


### Tanner Fiez, Lillian Ratliff. "Data-Analytics and Learning Methods for Modeling Spatio-Temporal Parking Demand," submitted to IEEE Transactions on Intelligent Transportation Systems (ITS) 2018.

### Chase Dowling, Tanner Fiez, Lillian Ratliff, Baosen Zhang. "Fusing Data Streams to Model Congestion Caused by Drivers Cruising for Curbside Parking," submitted to Nature Communications 2018.

### Tanner Fiez, Lillian Ratliff, Chase Dowling, Baosen Zhang. "Data-Driven Spatio-Temporal Modeling of Parking Demand," submitted to American Control Conference (ACC) 2018.

# Description
Management of curbside parking in cities has become increasingly important as these areas are expanding. To mitigate congestion, while meeting a city’s diverse needs, performance based pricing schemes have received a significant amount of attention. However, several recent studies suggest location, time of day, and awareness of policies are the primary factors that drive parking decisions. In light of this, we provide an extensive study of the spatio-temporal characteristics of parking demand. This work advances the understanding of where and when to set pricing policies, as well as how to target information and incentives to drivers looking to park. Harnessing data provided by the Seattle Department of Transportation, we develop a Gaussian mixture model based technique to identify zones with similar spatial demand as quantified by spatial autocorrelation. In support of this technique we provide a method based on the repeatability of our Gaussian mixture model to show demand for parking is consistent through time.

# Instructions
Code is contained in the code folder of the repository. The file descriptions
below explain the structure of the code and how to use the code to run various analysis.

# File Descriptions
##### load_sdot_utils.py
This file implements functions to pull and process parking transaction data from
the Seattle Department of Transportation (SDOT) API. The functions allow for
raw transaction data to be converted to minute by minute occupancy estimates
for each block-face in selected neighborhoods which can be aggregated to a desired
granularity. In create_data_example.py an example is provided to pull transaction
data for several months, create and write files containing minute by minute
occupancies for block-faces to folders separated by neighborhood, and use these
files to aggregate to hourly average occupancies.  

##### process_data.py:
This file contains a function to process hourly occupancies to average occupancies
over the time period in the data files. Block closures are also accounted for
in this processing. Additionally, GPS data is loaded for the block-faces. An
example of running this file is below. Using the function in this file enables
running all the analysis in the project.
```
import process_data

params = process_data.load_data(data_path, load_paths)
element_keys, loads, gps_loc, park_data, idx_to_day_hour, day_hour_to_idx = params
```

##### figure_functions.py:
This files provides several helper functions to visualize the data to gain
insights into the spatio-temporal properties of demand for parking. Some examples
using data from 2016 are given below.



<img src="/figs/belltown_division.png" width="400"/>

<img src="/figs/contours.png" width="400"/>

<img src="/figs/mixture_plot.png" width="400"/>

<img src="/figs/temporal_day_plots.png" width="1000"/>

##### mixture_animation.py:
This file provides functions to create a video animation of applying a
Gaussian mixture model.

##### gmm.py:

##### kmeans_utils.py:

##### map_overlay.py:

##### moran_auto.py:

##### run_analysis.py:

##### write_results.py:

# Dependencies
    python==2.7  
    seaborn==0.8.1          
    scipy==0.19.1     
    matplotlib==2.0.2    
    numpy==1.13.1    
    scikit_learn==0.19.0
    https://github.com/matplotlib/natgrid
