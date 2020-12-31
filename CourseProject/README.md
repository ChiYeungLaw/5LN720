# Project Assignment - Parsing Image Data: Captcha Recognition with Convolution Neural Network

To train a model:

```python
python train.py --train_path # the path of training data folder
				--label_path # the path of label file
				--dev_path # the path of development data folder
				--lr # learning rate
				--bs # batch size
				--num_epoch # the number of epoch
				--model_path # model save/load path
				--load_model # load pre-trained model?
				--cuda # use gpu?
				--layer # the number of CNN layers, 0 -> FNN model
				        # 1, 2, 3, 4, 5
```

To compute accuracy:

```python
python predict.py --test_path # the path of test data folder
				  --label_path # the path of label file
				  --bs # batch size
				  --model_path # model load path
				  --cuda # use gpu?
				  --layer # the number of CNN layers, 0 -> FNN model
				          # 1, 2, 3, 4, 5
```

