#include <iostream>
#include <vector>


#include "TMVA/DNN/DeepNet.h"
#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Architectures/Reference.h"
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

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

template <typename Architecture_t>
void train_mb(){
    
    using Scalar_t = typename Architecture_t::Scalar_t;
    using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
    using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
    
    /* For building a CNN one needs to define
    -  Input Layout :  number of channels (in this case = 1)  | image height | image width
    -  Batch Layout :  batch size | number of channels | image size = (height*width) */

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
    // TODO: fNet with batch size 1 for evaluating

    TConvLayer<Architecture_t> *convLayer = deepNet.AddConvLayer(4, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    convLayer->Initialize();
    TMaxPoolLayer<Architecture_t> *maxPool = deepNet.AddMaxPoolLayer(2, 2, 1, 1);
    TReshapeLayer<Architecture_t> *reshape1 = deepNet.AddReshapeLayer(0, 0, 0, 1);
    TDenseLayer<Architecture_t> *denseLayer1 = deepNet.AddDenseLayer(16, EActivationFunction::kRelu);
    denseLayer1->Initialize();
    TDenseLayer<Architecture_t> *denseLayer2 = deepNet.AddDenseLayer(10, EActivationFunction::kSigmoid);
    denseLayer2->Initialize();

    std::unique_ptr<DNN::VOptimizer<Architecture_t, Layer_t, DeepNet_t>> optimizer;
    optimizer = std::unique_ptr<DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>>(
                new DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>(
                deepNet, 0.001, 0.9, 0.999, 1e-7));

    
}