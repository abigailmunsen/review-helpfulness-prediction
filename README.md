# review-helpfulness-prediction
### README.md

The data necessary to run this program can be downloaded at http://jmcauley.ucsd.edu/data/amazon/.
We used the full home and kitchen data, which requires permission from the provider of the data. More information is at aforementioned link.

In order to run the program, simply run program.py. Edit the filenames provided if you're using different data.

The program will take a while to run. It will produce the following files:
- clean.json, the cleaned data
- features.json, the data with review features
- results.txt, the results
- withlogprice.png, expected vs. predicted values for model with log price
- withprice.png, expected vs. predicted values for model with raw price
- withoutall.png, expected vs. predicted values for model without any price information
