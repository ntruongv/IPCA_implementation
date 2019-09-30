# IPCA_implementation

The paper ``Characteristics are Covariances'' by Kelly-Pruitt-Su (2017) develops a statistical method called 
Instrumented Principal Component Analysis (IPCA) that allows for incorporating observation characteristics to 
estimate latent factors in a time-varying fashion. As we shall see, their approach could be essentially thought of as 
first projecting individual stock returns into managed-porfolios based on characteristics (which provides dimension reduction),
and then performing PCA on that.

We programmed IPCA from scratch in Python and applied it to 52-year data set with monthly observations of 36 characteristics 
of 10,000 stocks. 
