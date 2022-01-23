from ivu.data.training import TrainInputData


class TestInputData(TrainInputData):
    def __init__(self, df, train_type, class_column="class_label_index"):
        super().__init__(df, train_type, class_column)
