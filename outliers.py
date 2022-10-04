"""

This function finds outliers and returns them as pandas dataframes.

"""

class outliers:
    
    @time_cal
    def outliers(data: pd.DataFrame) -> pd.DataFrame:

        import pandas as pd
      
        index = []
        Q1    = []
        Q3    = []
        IQR   = []
        UIF   = []
        LIF   = []
        Max   = []
        Min   = []
        Std   = []
        Mdn   = []
        Mean  = []
      
        dataframe = {'Quantile 0.25'       : Q1,
                     'Quantile 0.75'       : Q3,
                     'Interquartile Range' : IQR,
                     "Upper Inner Fence"   : UIF,
                     'Lower Inner Fence'   : LIF,
                     "Maximum Values"      : Max,
                     "Minimum Values"      : Min,
                     "Standad Deviation"   : Std,
                     "Median"              : Mdn,
                     "Mean"                : Mean}
      
        for i in data.columns:
          
            index.append(data[i].name)
            Q1.append(data[i].quantile(0.25))
            Q3.append(data[i].quantile(0.75))
            IQR.append(data[i].quantile(0.75) - data[i].quantile(0.25))
            UIF.append(data[i].quantile(0.75) + 1.5 * (data[i].quantile(0.75) - data[i].quantile(0.25)))
            LIF.append(data[i].quantile(0.25) - 1.5 * (data[i].quantile(0.75) - data[i].quantile(0.25)))
            Max.append(data[i].max())
            Min.append(data[i].min())
            Std.append(data[i].std())
            Mdn.append(data[i].median())
            Mean.append(data[i].mean())
          
        table = pd.DataFrame(data = dataframe, index = index)
      
        return table
      
      
# EXAMPLE

"""

# Example Data

data = sns.load_dataset('tips')
data = data.drop([ "sex","smoker","day","time"], axis=1)

outliers.outliers(data)

# other

help(outliers.outliers)

"""
