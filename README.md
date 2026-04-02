# Scripts for running GeoDES jobs on CHTC

The experiment for each specific hyperparam / hyperparam combo is placed in its own respective directory.

### Naming conventions
Inside that directory, file naming convention is:
- 2D train jobs start with '2d_' followed by hyperparam abbreviated name (eg 'bs' for batch size) followed by the hyperparam choice (eg 4 for batch size of 4). Altogether '2d_bs4'
- 3D train jobs:
    - If the experiment concerns the 3D stage exclusively (e.g. temporal correlation, where no experiments are run to vary the 2D train process), use the same naming convention like 2D but prepend with '3d_' instead. Eg, '3d_tcorr5' for tempoal correlation of .5
    - If the experiment concerns both 2D and 3D stages, list the 2d number and then 3d number, separated by a period. Eg '3d_bs1.4' for 2D batch size was 1 and 3D batch size is 4.

### Multi-user considerations
Make sure you change the location of the source code geodes.tar.gz, so it pulls it from your own home directory
