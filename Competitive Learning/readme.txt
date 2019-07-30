This document conatins details concerning the processes required to reproduce
the results as presented in the report document.

To re-create the results for a specific technique, open the appropriately named file:

e.g. Opening and executing noise.py will recreate the results that are found in Appendix 
figur 10 for the applicaton of noise alongside the competitive learning algorithm.

This loops over the process 10 times and outputs the number of dead units for 10 different seed
positions from a list where the index of the list corresponds to the seed position.

e.g. index 0 contains the number of dead units from running the algorithm with seed 0.

---
To see a more detailed version of the process, open the "Assignment1.ipynb" file which contains
the full code including the plotting of graphs and figures. These include:

1. A frequency histogram which shows the number of times each neuron has fired
2. A correlation heatmap of the correlation between prototypes.
3. A visualisation of all of the output prototypes.
4. A graph showing the mean weight change as a function of time.

