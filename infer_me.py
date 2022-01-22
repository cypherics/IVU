import sys

from ivu.inference import Inference


INFER_FOR = ["normalized_distance_matrix", "distance_matrix", "normalized_key_points"]


def _run(infer_for, model_pth, data_dir, save_dir):
    infer_data = Inference.init_with_parameters(
        model_pth=model_pth,
        inference_parameters={"infer_for": infer_for},
        pose_estimator_parameters={
            "pose_estimator_complexity": 1,
            "use_pose_estimator_over_static_image": True,
        },
        video_parameters={"frame_height": -1, "frame_width": -1, "stride": 64},
        save_dir=save_dir,
        data_dir=data_dir,
    )
    infer_data.run()


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
