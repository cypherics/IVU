import sys

from ivu.trainer import Trainer

# TRAIN_PTH = r"datasets/train_key_points.pickle"
# TEST_PTH = r"datasets/test_key_points.pickle"


def temporal_kar_with_normalized_distance_matrix(conf_pth):
    temporal_kar_trainer_with_normalized_distance_matrix = (
        Trainer.train_with_normalized_distance_matrix(
            train_pth=TRAIN_PTH,
            test_pth=TEST_PTH,
            conf_pth=conf_pth,
        )
    )

    temporal_kar_trainer_with_normalized_distance_matrix.start_training()


def temporal_kar_with_distance_matrix(conf_pth):
    temporal_kar_trainer_with_distance_matrix = Trainer.train_with_distance_matrix(
        train_pth=TRAIN_PTH,
        test_pth=TEST_PTH,
        conf_pth=conf_pth,
    )

    temporal_kar_trainer_with_distance_matrix.start_training()


def temporal_with_normalized_key_points(conf_pth):
    temporal_kar_trainer_with_normalized_key_points = (
        Trainer.train_with_normalized_key_points(
            train_pth=TRAIN_PTH,
            test_pth=TEST_PTH,
            conf_pth=conf_pth,
        )
    )

    temporal_kar_trainer_with_normalized_key_points.start_training()


def lstm_with_normalized_key_points(conf_pth):
    lstm_kar_trainer_with_normalized_key_points = (
        Trainer.train_with_normalized_key_points(
            train_pth=TRAIN_PTH,
            test_pth=TEST_PTH,
            conf_pth=conf_pth,
        )
    )

    lstm_kar_trainer_with_normalized_key_points.start_training()


def lstm_with_normalized_distance_matrix(conf_pth):
    lstm_kar_trainer_with_normalized_distance_matrix = (
        Trainer.train_with_normalized_distance_matrix(
            train_pth=TRAIN_PTH,
            test_pth=TEST_PTH,
            conf_pth=conf_pth,
        )
    )

    lstm_kar_trainer_with_normalized_distance_matrix.start_training()


def lstm_with_distance_matrix(conf_pth):
    lstm_kar_trainer_with_distance_matrix = Trainer.train_with_distance_matrix(
        train_pth=TRAIN_PTH,
        test_pth=TEST_PTH,
        conf_pth=conf_pth,
    )

    lstm_kar_trainer_with_distance_matrix.start_training()


def _run_experiments():
    temporal_kar_with_normalized_distance_matrix(
        r"config/temporal_with_distance_matrix.yaml"
    )
    # temporal_kar_with_distance_matrix(r"config/temporal_with_distance_matrix.yaml")
    # temporal_with_normalized_key_points(
    #     r"config/temporal_with_normalized_key_points.yaml"
    # )
    #
    # lstm_with_normalized_key_points(r"config/lstm_kar_normalized_key_points.yaml")
    # lstm_with_normalized_distance_matrix(r"config/lstm_kar_distance_matrix.yaml")
    # lstm_with_distance_matrix(r"config/lstm_kar_distance_matrix.yaml")


if __name__ == "__main__":
    TRAIN_PTH = sys.argv[1]
    TEST_PTH = sys.argv[2]

    _run_experiments()
