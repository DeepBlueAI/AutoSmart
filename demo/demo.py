import auto_smart

info = auto_smart.read_info("data")
train_data,train_label = auto_smart.read_train("data",info)
test_data = auto_smart.read_test("data",info)
auto_smart.train_and_predict(train_data,train_label,info,test_data)

