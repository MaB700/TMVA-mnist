import ROOT

from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, MaxPooling2D, Flatten

# Setup TMVA
ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()
ROOT.EnableImplicitMT(4)

output = ROOT.TFile.Open("TMVA.root", "RECREATE")
factory = ROOT.TMVA.Factory(
    "MNIST", output,
    "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug")

# Load data
dataloader = ROOT.TMVA.DataLoader("dataset")

data = ROOT.TFile.Open("mnist0.root")

dataloader.AddRegressionTree(data.Get("train"), 1.0)

for i in range(28 * 28 * 1):
    dataloader.AddVariable("x[{}]".format(i), "x_{}".format(i), "", "F", 0.0, 1.0)

for i in range(10):
    dataloader.AddTarget("y[{}]".format(i), "y_{}".format(i), "", 0.0, 1.0)

dataloader.PrepareTrainingAndTestTree(
    ROOT.TCut(""), "SplitMode=Random:NormMode=None:V:!Correlations:!CalcCorrelations")

# Define model
#TODO: batchLayout needed?
inputLayoutString = "InputLayout=1|28|28:"
layoutString = "Layout=CONV|4|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|16|RELU,DENSE|10|IDENTITY:"
trainingString = "TrainingStrategy=MaxEpochs=15,BatchSize=100,LearningRate=1e-3:"
cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug"
options = inputLayoutString + layoutString + trainingString + cnnOptions
print(options) 

# Book methods

factory.BookMethod(dataloader, ROOT.TMVA.Types.kDL, "tmvaDL",
                   options)

# Run training, test and evaluation
factory.TrainAllMethods()
#factory.TestAllMethods()
#factory.EvaluateAllMethods()
