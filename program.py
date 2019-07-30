import pandas as pd

import processes.cleaner as cln
import processes.features as feat
import processes.neuralnet as nn

def main():
    #df = cln.create('meta_Home_and_Kitchen.json.gz', 'reviews_Home_and_Kitchen.json.gz')
    #df = feat.get_features(df)
    df = pd.read_json("features.json")
    nn.run(df)

main()
