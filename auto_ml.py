# in progress

from __future__ import annotations

import decorators

class ML:

  from sklearn.linear_model import (
      LinearRegression,
      Ridge
      )

  from sklearn.preprocessing import (
      StandardScaler,
      MinMaxScaler,
      normalize,
      Normalizer,LabelEncoder,
      scale,
      PowerTransformer,
      KBinsDiscretizer
      )

  from sklearn.model_selection import (
      train_test_split, 
      GridSearchCV,
      cross_val_score,
      cross_val_predict,
      RepeatedStratifiedKFold, 
      RepeatedKFold,
      KFold
      )

  import pandas as pd
  
  import numpy as np 

  @time_cal
  def ml(      
         data      : DataFrame, 
         y         : str, 
         X         : str, 
         one_hot   : list        = False, 
         lof       : list        = False, 
         scaler    : object      = [StandardScaler()], 
         test_size : float       = 0.20,
         ml_type   : str  | list = [LinearRegression(n_jobs=-1)],
         grids     : bool | str  = False,
         cv        : bool | str  = False,
         ensemble  : bool        = False
         ):
    
        # DATA PARAMETER
        # checking data parameter's type

        if type(data) != pd.core.frame.DataFrame:
          raise TypeError ("\n 'Data parameter must be in DataFrame format !!!' \n ")
        else:
          pass

        # cheking if there is ID etc columns in data

        data_columns_ID = []

        for i in data.columns:
          data_columns_ID.append(i)

        ID_list = ["ID","id","members","Member"]

        for i in ID_list:
          if i in data_columns_ID:
            raise IndexError ("Please drop " + str(i) + " columns from data !!!")
          else:
            pass

        # y PARAMETER
        # checking y parameter's type

        if type(y) != str:
          raise ValueError ("'y parameter must be string format !!!' ")
        else:
          pass
        
        # checking if parameter y exists in data

        data_columns_y = []

        for i in data.columns:
          data_columns_y.append(i)
          
        # print(data_columns)

        if y in data_columns_y:
            pass
        else:
            raise ValueError ("'y parameter must be in Data Columns !!!' ")

        # X PARAMETER
        # checking X parameter's type

        if type(X) != str:
          raise ValueError ("'X parameter must be string format !!!' ")
        else:
          pass
        
        # checking if parameter X exists in data

        data_columns_X = []

        for i in data.columns:
          data_columns_X.append(i)
          
        # print(data_columns_X)

        if X in data_columns_X:
            print("one hot başlangıcı")
            pass
        else:
            raise ValueError ("'X parameter must be in Data Columns !!!' ")

        # ONE-HOT ENCODING

        if one_hot != False:
          if type(one_hot) != list:
            raise ValueError ("'one_hot parameter must be list format !!!' ")
          else:
            print("başlangıç")
            display(data)
            data = pd.get_dummies(data, columns = one_hot , drop_first = True)
            display(data)
            print("1. Aşama")
        else:
          pass

        # LOCAL OUTLIER FACTOR

        if lof != False:
          if type(lof) != list:
             raise ValueError("lof parameters must be list type")
          else:
            pass
            #LocalOutlierFactor(n_neighbors = n_neighbors)

        # DATA STANDARDIZATION and SPLITTING TEST/TRAIN
        # Data Standartlaştırma da parametre / argüman sözlük formatında isteyebilirsin.
        # çünkü fonksiyonların kendi parametreleri var. PowerTransformer(method='yeo-johnson') gibi
        
        if type(test_size) != float:
          raise ValueError ("'test_size parameter must be float format !!!' ")
        else:
          pass

        if scaler == False:

          y = data[[y]]

          X = data.drop(y, axis=1)

          X_train, X_test, y_train, y_test = train_test_split(
              X, 
              y, 
              test_size = test_size,
              random_state = 17)
        else:
          if type(scaler) != list:
            raise ValueError ("'scaler parameter must be list format !!!' ")
          else:
            for i,j in enumerate(scaler):
              if i > 0:
                raise IndexError ("'must be add only one scaler parameter!!!'")
              else:
                y = data[[y]]
                y_scaled =  scaler[0].fit_transform(y)
                X = data.drop(y, axis=1)
                X_scaled = scaler[0].fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, 
                    y_scaled, 
                    test_size = test_size,
                    random_state = 17)
                
                # normalize ve scaler argüman istiyor.

        """        
        
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        
        """

        # ml_type PARAMETER

        """

        RAll
        ---------------------
        RAll is means: 'All Regression models (RAll) will fitting' 
        Note: If you choise RAll method all machine learning models will fitting with default parameters

        CAll
        ---------------------
          CAll is means: 'All Classification models (CAll) will fitting'
        Note: If you choise CAll method all machine learning models will fitting with default parameters

        # Also you can choise spesified algoritms in a list

        """

        if ml_type == "RAll":

          # Regression ML Models

          KNNR    = KNeighborsRegressor            ()

          HGBR    = HistGradientBoostingRegressor  (random_state = 17)

          LGBMR   = LGBM.LGBMRegressor             (random_state = 17)

          ETR     = ExtraTreesRegressor            (random_state = 17)

          RFR     = RandomForestRegressor          (random_state = 17)

          GBR     = GradientBoostingRegressor      (random_state = 17)

          PR      = linear_model.PoissonRegressor  ()

          BAGR    = BaggingRegressor               (ETR, random_state = 17)

          LCV     = LassoCV                        (random_state = 17)

          LL      = linear_model.LassoLars         (random_state = 17)

          BR      = linear_model.BayesianRidge     ()

          RCV     = RidgeCV                        ()

          LLCV    = LassoLarsCV                    ()

          LARSCV  = LarsCV                         ()

          R       = Ridge                          (random_state = 17)

          TT      = TransformedTargetRegressor     (regressor=LinearRegression())

          LARS    = linear_model.Lars              (random_state = 17)

          LR      = LinearRegression               ()

          LLIC    = linear_model.LassoLarsIC       ()

          LASSO   = linear_model.Lasso             (random_state = 17)

          SGDR    = make_pipeline                  (StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3,random_state = 17))

          EN      = ElasticNet                     (random_state = 17)

          HR      = HuberRegressor                 ()

          PAR     = PassiveAggressiveRegressor     (random_state = 17)

          GR      = linear_model.GammaRegressor    ()

          TR      = linear_model.TweedieRegressor  ()

          OMP     = OrthogonalMatchingPursuit      ()

          RANSACR = RANSACRegressor                (random_state = 17)

          OMPCV   = OrthogonalMatchingPursuitCV    ()

          ABR     = AdaBoostRegressor              (random_state = 17)

          ENCV    = ElasticNetCV                   (random_state = 17)

          DR      = DummyRegressor                 ()

          NSVR    = make_pipeline                  (StandardScaler(), NuSVR())

          SVR_    = make_pipeline                  (StandardScaler(), SVR(C=1.0, epsilon=0.2))

          MLPR    = MLPRegressor                   (random_state = 17)

          KR      = KernelRidge                    ()

          LSVR    = make_pipeline                  (StandardScaler(),LinearSVR(random_state = 17))

          GPR     = GaussianProcessRegressor       (kernel=DotProduct() + WhiteKernel(),random_state = 17)


          ML_RALL = [
              KNNR,
              HGBR,
              LGBMR,
              ETR,
              RFR,
              GBR,
              PR,
              BAGR,
              LCV,
              LL,
              BR,
              RCV,
              LLCV,
              LARSCV,
              R,
              TT,
              LARS,
              LR,
              LLIC,
              LASSO,
              SGDR,
              EN,
              HR,
              PAR,
              GR,
              TR,
              OMP,
              RANSACR,
              OMPCV,
              ABR,
              ENCV,
              DR,
              NSVR,
              SVR_,
              MLPR,
              KR,
              LSVR,
              GPR
              ]
          
          for i in ML_RALL:
            i.fit(X_train, y_train)
            y_pred = i.predict(X_test)
            print (np.sqrt(mean_squared_error(y_test,y_pred)))

        elif ml_type == "CAll":

          # Classification ML Models

          ABC   = AdaBoostClassifier(n_estimators=100, random_state=0)

          QDA   = QuadraticDiscriminantAnalysis()

          SVC_  = make_pipeline(StandardScaler(), SVC(gamma='auto'))

          NSVC  = make_pipeline(StandardScaler(), NuSVC())

          LGBMC = LGBM.LGBMClassifier()

          LR_    = LogisticRegression(random_state=0)

          LP    = LabelPropagation()

          LS    = LabelSpreading()

          RFC   = RandomForestClassifier(max_depth=2, random_state=0)

          NC    = NearestCentroid()

          KNNC  = KNeighborsClassifier(n_neighbors=3)

          ETC   = ExtraTreesClassifier(n_estimators=100, random_state=0)

          CCCV  = CalibratedClassifierCV(base_estimator=GaussianNB(), cv=3)

          LSVC  = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))

          XGBC  = XGB.XGBClassifier()

          SGDC  = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))

          LDA   = LinearDiscriminantAnalysis()

          RC    = RidgeClassifier()

          RCCV  = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])

          GNB   = GaussianNB()

          BNB   = BernoulliNB()

          BC    = BaggingClassifier(ExtraTreeClassifier(random_state=0), random_state=0)

          DTC   = DecisionTreeClassifier(random_state=0)

          P     = Perceptron(tol=1e-3, random_state=0)

          PAC   = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)

          DC    = DummyClassifier(strategy="most_frequent")

          ML_CALL = [
              ABC,
              QDA,
              SVC_,  
              NSVC,
              LGBMC, 
              LR_,
              LP,
              LS,   
              RFC,   
              NC,  
              KNNC,  
              ETC,
              CCCV,  
              LSVC, 
              XGBC, 
              SGDC,
              LDA, 
              RC, 
              RCCV,  
              GNB, 
              BNB,  
              BC,  
              DTC,   
              P,  
              PAC,   
              DC
              ]
          
          for i in ML_CALL:
            i.fit(X_train, y_train)
            y_pred = i.predict(X_test)
            print (np.sqrt(mean_squared_error(y_test,y_pred)))

        else:
            if type(ml_type) != list:
                raise ValueError ("'ml_type parameter must be list format !!!' ")
            else: 
              for i,j in enumerate(ml_type):
                if i > 0:
                  j.fit(X_train, y_train)
                  y_pred = j.predict(X_test)
                  result_y = (np.sqrt(mean_squared_error(y_test, y_pred)))
                  return result_y
                else:
                  alg = ml_type[0].fit(X_train, y_train)
                  y_pred = alg.predict(X_test)
                  result_x =  (np.sqrt(mean_squared_error(y_test, y_pred)))
                  return result_x
