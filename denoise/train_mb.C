#include <iostream>
#include <vector>


#include "TMVA/DNN/DeepNet.h"
#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/Tools.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DLMinimizers.h"
#include "TMVA/DNN/Optimizer.h"
#include "TMVA/DNN/SGD.h"
#include "TMVA/DNN/Adam.h"
#include "TMVA/DNN/Adagrad.h"
#include "TMVA/DNN/RMSProp.h"
#include "TMVA/DNN/Adadelta.h"
#include "TMVA/Timer.h"
#include "TMVA/Tools.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TRandom2.h"
#include "TStopwatch.h"

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

std::vector<TMVA::Event *> loadEvents();
TMVA::DataSetInfo &getDataSetInfo();


void train_mb(){
    
    using Architecture_t = TMVA::DNN::TCpu<Float_t>;
    using Scalar_t = typename Architecture_t::Scalar_t;
    using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
    using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
    using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

    Architecture_t::SetRandomSeed(0);

    // load data
    size_t nTrainingSamples(50000);
    size_t nValidationSamples(10000);
    const std::vector<TMVA::Event *> allData = loadEvents();
    const std::vector<TMVA::Event *> eventCollectionTraining{allData.begin(), allData.begin() + nTrainingSamples};
    const std::vector<TMVA::Event *> eventCollectionValidation{allData.begin() + nTrainingSamples, allData.end()};

    
    
    /* For building a CNN one needs to define
    -  Input Layout :  number of channels (in this case = 1)  | image height | image width
    -  Batch Layout :  batch size | number of channels | image size = (height*width) */

    size_t nThreads = 1; 
    size_t batchSize = 100;
    size_t inputDepth  = 1;
    size_t inputHeight = 28;
    size_t inputWidth  = 28;
    size_t batchDepth  = batchSize; 
    size_t batchHeight = 1;
    size_t batchWidth  = inputHeight*inputWidth; 
    ELossFunction J    = ELossFunction::kMeanSquaredError;
    EInitialization I  = EInitialization::kUniform;
    ERegularization R    = ERegularization::kNone;
    EOptimizer O         = EOptimizer::kAdam;
    Scalar_t weightDecay = 0; //FIXME:
    
    DeepNet_t deepNet(batchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R, weightDecay);
    DeepNet_t fNet(1, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R, weightDecay);

    TConvLayer<Architecture_t> *convLayer = deepNet.AddConvLayer(4, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    convLayer->Initialize();
    TMaxPoolLayer<Architecture_t> *maxPool = deepNet.AddMaxPoolLayer(2, 2, 1, 1);
    TReshapeLayer<Architecture_t> *reshape1 = deepNet.AddReshapeLayer(0, 0, 0, 1);
    TDenseLayer<Architecture_t> *denseLayer1 = deepNet.AddDenseLayer(16, EActivationFunction::kRelu);
    denseLayer1->Initialize();
    TDenseLayer<Architecture_t> *denseLayer2 = deepNet.AddDenseLayer(10, EActivationFunction::kSigmoid);
    denseLayer2->Initialize();

    // Loading the training and validation datasets
    TMVAInput_t trainingTuple = std::tie(eventCollectionTraining, getDataSetInfo()); //TODO:
    TensorDataLoader_t trainingData(trainingTuple, nTrainingSamples, batchSize,
                                    {inputDepth, inputHeight, inputWidth},
                                    {deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth()} ,
                                    deepNet.GetOutputWidth(), nThreads);

    TMVAInput_t validationTuple = std::tie(eventCollectionValidation, getDataSetInfo());
    TensorDataLoader_t validationData(validationTuple, nValidationSamples, batchSize,
                                    {inputDepth, inputHeight, inputWidth},
                                    { deepNet.GetBatchDepth(),deepNet.GetBatchHeight(), deepNet.GetBatchWidth()} ,
                                    deepNet.GetOutputWidth(), nThreads);

    
    
    
    std::unique_ptr<TMVA::DNN::VOptimizer<Architecture_t, Layer_t, DeepNet_t>> optimizer;
    optimizer = std::unique_ptr<TMVA::DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>>(
                new TMVA::DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>(
                deepNet, 0.001, 0.9, 0.999, 1e-7));

    size_t batchesInEpoch = nTrainingSamples / deepNet.GetBatchSize();
    size_t epoch(0);
    size_t maxEpoch(5);
    TMVA::RandomGenerator<TRandom3> rng(123); 
    while(epoch < maxEpoch){
        TStopwatch timer;
        timer.Start();
        trainingData.Shuffle(rng);
        
        for (size_t i = 0; i < batchesInEpoch; ++i ){
            auto my_batch = trainingData.GetTensorBatch();
            deepNet.Forward(my_batch.GetInput(), true);
            deepNet.Backward(my_batch.GetInput(), my_batch.GetOutput(), my_batch.GetWeights());
            optimizer->IncrementGlobalStep();
            optimizer->Step();
        }
        
        // Compute training error.
        Double_t trainError = 0.;
        for (auto batch : trainingData){
            auto inputTensor = batch.GetInput();
            auto outputMatrix = batch.GetOutput();
            auto weights = batch.GetWeights();
            trainError += deepNet.Loss(inputTensor, outputMatrix, weights, false, false);
        }
        trainError /= (Double_t)(nTrainingSamples / batchSize);
        
        // Compute validation error.
        Double_t valError = 0.;
        for (auto batch : validationData){
            auto inputTensor = batch.GetInput();
            auto outputMatrix = batch.GetOutput();
            auto weights = batch.GetWeights();
            valError += deepNet.Loss(inputTensor, outputMatrix, weights, false, false);
        }
        valError /= (Double_t)(nValidationSamples / batchSize);
        timer.Stop();
        Double_t dt = timer.RealTime();
    
        epoch++;
    }


}


std::vector<TMVA::Event *> loadEvents(){ //target is denoisy input

    TStopwatch timer;
    timer.Start();
    std::cout << "loading Events from file ... \n" ; 
    TFile *file = TFile::Open("mnist.root"); 
    TTree *tree = (TTree*)file->Get("train"); 
    Float_t x0[784];
    tree->SetBranchAddress("x", x0);
    std::vector<TMVA::Event*> allData;
    
    TRandom2 *rand = new TRandom2(123); //fixed seed
    
    Long64_t nofEntries = tree->GetEntries();
    for(Long64_t i=0; i < nofEntries; i++){
        tree->GetEntry(i);
        std::vector<Float_t> input; 
        std::vector<Float_t> target; 
        
        for(int j=0; j < 784; j++){
            target.push_back(x0[j]);
            //add noise
            Float_t tar = x0[j] + rand->Rndm() * 0.2;
            if(tar > 1.) tar = 1. ;
            input.push_back(tar);
        }
        

        TMVA::Event* ev = new TMVA::Event(input, target);
        allData.push_back(ev); 
    }
    timer.Stop();
    Double_t dt = timer.RealTime();
    std::cout << "loading finished ! \n" ; 
    std::cout << "time for loading Events  " << dt << "s\n";  
    return allData;
}

TMVA::DataSetInfo &getDataSetInfo(){
    TMVA::DataSetInfo* dsi;
    for(int i{0}; i < 28*28; i++){
        TString in1; in1.Form("in%d", i);
        TString in2; in2.Form("input pixel %d", i);
        dsi->AddVariable(in1, in2, "", 0., 1.);

        TString out1; out1.Form("out%d", i);
        TString out2; out2.Form("output pixel %d", i);
        dsi->AddTarget(out1, out2, "", 0., 1.);
    
    return *dsi;
}