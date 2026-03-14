**程序所需安装的库位于`requirements.md`,**

**训练程序**
 - 运行`train_CNN.py`开始进行CNN模型的训练
 - 运行`train_LSTM.py`开始进行LSTM模型的训练
 - 运行`train_MLP.py`开始进行MLP模型的训练
 - 训练好的结果将保存在`saved_models`文件夹中；grid search的结果会保存至对应的.csv文件
 - `saved_models_res`是已经训练好的模型，`grid_search_results`是相应训练的时候保存的grid search结果
  
**测试程序**
直接运行程序，所测试的模型都是在本地跑过保存好的结果；如需要测试重新训练得到的模型，请修改代码中的路径`model_path = "saved_models_res/model_fs-(2, 3, 4)_dr-0.3_nf-64_lr-0.001.pt"` 为相应路径
 - 运行`testing_CNN.py`测试由accuracy评测出的最好CNN模型
 - 运行`testing_LSTM.py`测试由accuracy评测出的最好LSTM模型
 - 运行`testing_MLP.py`测试由accuracy评测出的最好MLP模型

**文件说明**
- `saved_models_res`目录下是实验者在本地运行得到的模型结果
- `grid_search_results`是上述训练的时候保存的grid search结果
- `data`目录是训练数据集，需要自行提供
- `wiki_word2vec_50.bin`是转换词语到词向量的bin文件，需要自行提供
- `.py`是上述训练或测试程序
- `Readme.md`说明文档
- `requirements.md`所需环境配置说明


**Required libraries and environment setup are listed in `requirements.md`.**

**Training**
- Run `train_CNN.py` to train the CNN model.
- Run `train_LSTM.py` to train the LSTM model.
- Run `train_MLP.py` to train the MLP model.
- The trained models will be saved in the `saved_models` folder, and the grid search results will be saved in the corresponding `.csv` files.
- `saved_models_res` contains pre-trained models, and `grid_search_results` contains the corresponding grid search results saved during training.

**Testing**  
Run the testing scripts directly. The models being tested are pre-trained models that have already been run and saved locally. If you want to test newly trained models, please modify the path in the code  
`model_path = "saved_models_res/model_fs-(2, 3, 4)_dr-0.3_nf-64_lr-0.001.pt"`  
to the corresponding model path.

- Run `testing_CNN.py` to test the best CNN model selected by accuracy.
- Run `testing_LSTM.py` to test the best LSTM model selected by accuracy.
- Run `testing_MLP.py` to test the best MLP model selected by accuracy.

**File Description**
- The `saved_models_res` directory contains model results obtained by the author through local experiments.
- `grid_search_results` contains the grid search results saved during the training process.
- The `data` directory contains the training dataset, which must be provided separately.
- `wiki_word2vec_50.bin` is the binary file used to convert words into word vectors, and must be provided separately.
- The `.py` files are the training and testing scripts described above.
- `Readme.md` is the documentation file.
- `requirements.md` contains the required environment configuration.



Direcroty Structures / 文件目录
```
.
├── data/                           # 数据集目录 / dataset directory
│   ├── train.txt                   # 训练集 / training set
│   ├── test.txt                    # 测试集 / test set
│   └── validation.txt              # 验证集 / validation set
├── grid_search_results/            # 网格搜索结果 / grid search results
├── saved_models_res/               # 已训练好的模型结果 / saved model results
├── Readme.md                       # 项目说明文档 / project documentation
├── requirements.md                 # 环境与依赖说明 / environment and dependency notes
├── testing_CNN.py                  # CNN 模型测试脚本 / CNN testing script
├── testing_LSTM.py                 # LSTM 模型测试脚本 / LSTM testing script
├── testing_MLP.py                  # MLP 模型测试脚本 / MLP testing script
├── train_CNN.py                    # CNN 模型训练脚本 / CNN training script
├── train_LSTM.py                   # LSTM 模型训练脚本 / LSTM training script
├── train_MLP.py                    # MLP 模型训练脚本 / MLP training script
└── wiki_word2vec_50.bin            # 词向量二进制文件 / binary word embedding file
```
