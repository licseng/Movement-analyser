# Movement-analyser

Statistical analysis script for movement data, extracted in json-format with [OpenPose (2d)](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). OpenPose or 3d-pose-baseline needs to be run externally, this script works with their output files.

The script does the following:
- Imputing missing data
- Applying Savitzky-Golay filter
- Descriptive statistics extraction with additional filters
- Saving results into 'metrics.csv'
- Saving plot of signed dispositions of every frame

## In 2d mode
    $ python stat_analysis.py 2d <2Djson_dirname> <video_name>
Script works with the output directory of OpenPose which contains json files for every video frame with 25 body points
Every point has 3 variables: x coordinate, y coordinate, detection confidence
    
## In 3d mode
Stat_analysis.py works with the output json file of 3d-pose-baseline, a pretrained model that lifts OpenPose's output 2d coordinates to 3d space.
However the lifting algorithm accepts only cleaned json files, so this script has to be run first:

    $ python preparing_2Djsons.py <2Djson_dirname> <clean2Djson_dirname>

Then the statistical analysis:

    $ python stat_analysis.py 3d <3Djson_filename>

Every 3d-pose-baseline output point has 3 variables: x coordinate, y coordinate, z coordinate

## Mind that
- the extracted statistics are **NOT validated properly**! More smoothing algorithms (for time series data) should be tested.
- during comparison of videos' statistics, fps of the examined videos matter! 
- with different versions of OpenPose and 3d-pose-baseline the examined bodypoints and their index could change! 

## Example
*video1* - 2d json data dict

*example_clean* - preprocessed 2d json data dict for 3d lifting 

*3d_data_example.json* - 3d json data

![Image2d](/Signed_avg_disp_2d_neni_exp03.png)
![Image3d](/Signed_avg_disp_3d.png)

**Videos are not uploaded due to privacy issues!**
