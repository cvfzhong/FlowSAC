# FlowSAC for Robust Flow Estimation

This is a reference implementation of **FlowSAC**, a robust method for estimating dense flow from a set of sparse matches that may be contaminated with significant outliers.

This repository can be used to reproduce the main experimental results in the paper. A copy of raw detailed results can be found at `./data/results-in-the-paper/`. Please visit our [project page](...) for the required data.


## Installation
The code is tested with Ubuntu 14.04 and [OpenCV 3.3.1](https://github.com/opencv/opencv/releases/tag/3.3.1). Slightly different results may be produced with different system configurations or other versions of OpenCV.

OpenCV is assumed to be installed at `/usr/local/include`, please modify the makefile accordingly if it is not. Other required packages are included at `./extern/`, including the implementation of [EpicFlow](https://thoth.inrialpes.fr/src/epicflow/) and [RicFlow](https://github.com/YinlinHu/Ric). 

The code can be easily built with the following commands:
```
# build EpicFlow, see instructions in extern/EpicFlow/README if any problem encountered.
cd extern/EpicFlow
make

# build RicFlow
cd extern/Ric-master
make

# build FlowSAC, some .o files of EpicFlow and RicFlow are required
cd src
make all
```
It will generate 3 command line tools in `./src/`: 
* `flowsac1`: processing a single flow pair with FlowSAC and FlowSAC-fast.
* `flowsac2`: processing a list of flow pairs with one of the four estimation methods (EpicFlow, RicFlow, FlowSAC and FlowSAC-fast), and can be used to reproduce the results of Table 1,2,3,5,7 in the paper.
* `flowsac3`: performing the robustness experiment of Table 6.

## Usages

### 1. Run for a singel flow pair 
For quick startup we have included an example in `./data/E1/`. You can run FlowSAC or FlowSAC-fast for it as:
```
cd ./data/E1
../../src/flowsac1 src.png tar.png match.txt ucm.png sed.edge outflow.flo sintel -gt gt.flo -L 3 -ucmT 0.1 -ransacT 2.6 -score SOD -fast -r 10
```
All unnamed arguments must be provided:
* `src.png tar.png` are the input image pair.
* `match.txt` contains the input sparse matches.
* `ucm.png` is the Ultrametric Contour Map(UCM) produced with [MCG](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/). It should be a CV_32FC1 image of the same size as `src.png`. The 32FC1 image data is encoded with 8UC4 in order to save as png file.
* `sed.edge` is the [SED](https://github.com/pdollar/edges) edge map required by the LA flow interpolation of EpicFlow.
* `outflow.flo`: the output flow file
* `sintel`: the dataset name used for determine the default parameters (`-ucmT -ransacT` , and some parameters of EpicFlow). It can only be `sintel` or `kitti`.

All named arguments are optinal:

* `-gt` : the groundtruth flow, if is provided, AEE accuracy will be computed and showed and the end.
* `-L` : the maximum number of progressive iterations (`default=3`)
* `-ucmT` : the UCM threshold to generate the initial segmentation (`default=0.1 for sintel and 0.5 for kitti`)
* `-ransacT` : the RANSAC threshold (`default=2.6 for sintel and 5.0 for kitti`)
* `-score` : the flow score measure. It can be SOD, SAD, SCT, SNCC (`default=SOD`)
* `-fast` : use the fast mode, which will be 2-5 times faster
* `-r` : the R parameter to determine the downsampling rate in the fast mode (`default=10`)

### 2. Run for a list of flow pairs 

`flowsac2` can be used to reproduce most of our experiments in the paper. Please first download the required data of Sintel and Kitti2012 from our [project page](...). For each flow pair we have provided `sed.edge`, `ucm.png`, and matches of DCFlow, FlowFields, CPM, and DiscreteFlow. The images and the groundtruth can be downloaded from the [Sintel](http://sintel.is.tue.mpg.de/downloads) and [Kitti2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow) websites.

For example, you can run FlowSAC-fast for Sintel-train and FlowFields matches as:
```
cd src
flowsac2 ../data/sintel-train/list-all.txt dcflow sintel -method flowsac -o results.txt -fast 
```
* `.../list-all.txt`: the list of flow pairs.
* `dcflow`: name of the match file. With the provided dataset, it can be `dcflow`, `flowfields`, `cpm` and `discreteflow`.
* `-method`: the estimation method (`flowsac`, `epicflow`, `ricflow`). FlowSAC-fast should be specified with the `-fast` option.
* `-o`: the file to output results. For each flow pair, the AEE accuracy and the timecost will be recorded.
* `-L -ucmT -ransacT -score -r`: the same as `flowsac1`.

### 3. The robustness experiment

To run the robustness experiment (Table 6 in the paper), please first download [sintel-train-matchx](...), which contains the data to generate matches of different outlier ratio, for the matching methods of DCFlow, FlowFields1 (one-way crosschecking), FlowFields2(two-way crosschecking), CPM, and DiscreteFlow. Put the decompressed `matchx` folder to `./data/sintel-train/`. The list file `sintel-train/list-hard.txt` is the Sintel-hard dataset as described in the paper.

As an example, you can run the experiment for DCFlow matches as:
```
cd src
flowsac3 ../data/sintel-train/list-hard.txt dcflow sintel -o results.txt
```
For other matching methods, just replace `dcflow` with `flowfields1`, `flowfields2`, `cpm` or `discreteflow`.

For each flow pair, it calls the 4 estimation methods (EpicFlow, RicFlow, FlowSAC, FlowSAC-fast) for the 10 different crosschecking thresholds (3, 5, 8, 11, 15, 20, 25, 30, 40, 50), which will take serveral minutes to be finished. The results will finally be saved to `-o results.txt`.







