from saint_main import train_saint
from ARM_Net import train_arm_net
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd


train_data_path = r"data\splits\fold0train.csv"
test_data_path = r"data\splits\fold0test.csv"

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
y_train = train.iloc[:, -1].to_numpy()
x_train = train.drop(train.columns[-1],axis=1).to_numpy()
y_test = test.iloc[:, -1].to_numpy()
x_test = test.drop(test.columns[-1],axis=1).to_numpy()

clf = TabNetClassifier()  
clf.fit(
  x_train, y_train,
  eval_set=[(x_test, y_test)],
  max_epochs=3,
  eval_metric=["accuracy"]
)

saving_path_name = "./tabnet_model_test_1"
saved_filepath = clf.save_model(saving_path_name)

train_arm_net(train_data_dir=train_data_path, test_data_dir=test_data_path, repeat=1, max_epochs=3)
train_saint(data_path_train=train_data_path, data_path_test=test_data_path, epochs=3)