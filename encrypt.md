Protecting Core ML Models - Encrypt/Decrypt model

[Coreml: Model class has not been generated yet](https://stackoverflow.com/questions/46247611/coreml-model-class-has-not-been-generated-yet)

`target -> Build Setting -> CoreML Model Compiler - Code Generation -> CoreML Generated Model Inherits NSObject -> YES`

[Use model deployment and security with Core ML](https://developer.apple.com/videos/play/wwdc2020/10152/)

[Custom Encrypt/Decrypt model](https://heartbeat.comet.ml/protecting-core-ml-models-35218d93dcf7)

```
% xcrun coremlcompiler compile LARModel.mlpackage LAREncryptedModel.mlpackage --encrypt LARModel.mlmodelkey

LAREncryptedModel.mlpackage/LARModel.mlmodelc/coremldata.bin
```

Archived package: LAREncryptedModel.mlpackage
`encrytionInfo/coremldata.bin`

```
➜  encrypted ls
LARModel.mlmodelc   LARModel.mlmodelkey
➜  encrypted cd LARModel.mlmodelc 
➜  LARModel.mlmodelc ls
analytics      coremldata.bin encryptionInfo model.mil      weights
➜  LARModel.mlmodelc cd encryptionInfo 
➜  encryptionInfo ls
coremldata.bin
```

Cannot load LAREncryptedModel.mlpackage because:
There was a problem decoding this Core ML document
coremlcompiler: error: Failed to read model package at file:///.../LAREncryptedModel.mlpackage/Manifest.json

```
// Manifest.json
{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {
        "6CA91DC8-128F-4C3B-B60E-1C249EFB7152": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        },
        "7F47CB46-C996-4998-B88F-BABC28C75285": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        }
    },
    "rootModelIdentifier": "7F47CB46-C996-4998-B88F-BABC28C75285"
}
```
