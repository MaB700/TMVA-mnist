import ROOT

# load model from file
""" factory1 = ROOT.TMVA.Factory(
    "MNIST", 
    "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug")
factory1.BookMethodWeightfile(dataloader, ROOT.TMVA.Types.kDL, "model.xml")
model1 = factory1.GetMethod().GetDeepNet()
pred
prediction = Prediction( pred, , ) """