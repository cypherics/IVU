from ivu.inference import Inference

DATA_DIR = ""
SAVE_DIR = ""
MODEL_PTH = ""

INFER_FOR = ["normalized_distance_matrix", "distance_matrix", "normalized_key_points"]

infer_data = Inference.init_with_parameters(
    model_pth=MODEL_PTH,
    inference_parameters={"infer_for": INFER_FOR[2]},
    pose_estimator_parameters={
        "pose_estimator_complexity": 1,
        "use_pose_estimator_over_static_image": True,
    },
    video_parameters={"frame_height": -1, "frame_width": -1, "stride": 64},
    save_dir=SAVE_DIR,
    data_dir=DATA_DIR,
)
infer_data.run()
