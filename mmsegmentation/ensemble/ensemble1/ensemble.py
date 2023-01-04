import pandas as pd
import numpy as np
from tqdm import tqdm


def hard_voting_ensemble(out_path, *args):
    """
    out_path : output의 csv 경로와 이름 설정
    *args : out_put csv의 경로를 입력해주세요
    """
    input = [pd.read_csv(i) for i in args]
    submission = pd.DataFrame()
    submission["image_id"] = input[0]["image_id"]
    submission["PredictionString"] = np.NaN

    for i in tqdm(range(len(input[0]))):
        input_pred = [my_input["PredictionString"][i].split() for my_input in input]

        prediction = []
        for j in range(len(input_pred[0])):
            count = [out[j] for out in input_pred]
            cnt = []
            for k in range(len(input_pred)):
                cnt.append(count.count(count[k]))
            prediction.append(count[cnt.index(max(cnt))])
        submission["PredictionString"][i] = " ".join(str(z) for z in prediction)

    submission.to_csv(out_path, index=False)


if __name__ == "__main__":
    hard_voting_ensemble(
        "/opt/ml/input/code/mmsegmentation/ensemble/ensemble1/e_out.csv",
        "/opt/ml/input/code/submission/efficient_unet_best_model.csv",
        "/opt/ml/input/code/submission/gh_base.csv",
        "/opt/ml/input/code/submission/runs_2022-12-26_17-49-02.csv",
    )
