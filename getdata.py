##
from LULC import LULC
import os
import sys
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import boto3
from eolearn.core import (
    EOPatch,
    EOWorkflow,
    EOExecutor,
    linearly_connect_tasks,
    FeatureType,
    OverwritePermission,
    LoadTask,
    SaveTask
)
from eolearn.features import LinearInterpolationTask, SimpleFilterTask
import cv2

##
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASZDTXP65UX7LJPLQ"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


def get_patch_list():
    # Connect to S3 Bucket
    client = boto3.client(
        's3',
        aws_access_key_id="AKIASZDTXP65UX7LJPLQ",
        aws_secret_access_key="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",
        region_name='us-east-1')
    bucket = 'eo-learn.sentinel-hub.com'
    prefix = 'eopatches_slovenia_2019/'
    result = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    subfolder_list = []
    for o in result.get('CommonPrefixes'):
        subfolder_list.append(o.get('Prefix').replace("eopatches_slovenia_2019/", ""))

    # Marginal patches have too many "No Data" labels in masks while satellite images capture valid land over, which adds unnecessary noice to training process. Remove patches with too many "No Data" labels. Default ratio set at 0.1
    subfolder_list_clean = []
    for i in tqdm(range(len(subfolder_list))):
        eopatch = EOPatch.load(f"s3://eo-learn.sentinel-hub.com/eopatches_slovenia_2019/{subfolder_list[i]}",
                               features=(FeatureType.MASK_TIMELESS, "LULC"), lazy_loading=True)
        if np.sum(eopatch.mask_timeless['LULC'] == 0) / np.prod(eopatch.mask_timeless['LULC'].shape) <= 0.1:
            os.makedirs(f"./datasets/{subfolder_list[i]}", exist_ok=True)
            subfolder_list_clean.append(subfolder_list[i])
        del eopatch
    return subfolder_list_clean


##
def linear_task():
    # Loader
    load = LoadTask("s3://eo-learn.sentinel-hub.com/eopatches_slovenia_2019/", features=[
        (FeatureType.DATA, 'BANDS'),
        (FeatureType.MASK, 'IS_VALID'),
        (FeatureType.MASK_TIMELESS, "LULC"),
        FeatureType.TIMESTAMP
    ])

    # Valid pixel filter
    class ValidDataFractionPredicate:
        def __init__(self, threshold):
            self.threshold = threshold

        def __call__(self, array):
            coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
            return coverage > self.threshold

    valid_data_predicate = ValidDataFractionPredicate(0.8)
    filter_task = SimpleFilterTask((FeatureType.MASK, "IS_VALID"), valid_data_predicate)

    # Set time intervel to 15 days so number of sampled images will be ceil(365/15) = 25
    resampled_range = ("2019-01-01", "2019-12-31", 15)
    linear_interpolation = LinearInterpolationTask((FeatureType.DATA, "BANDS"),
                                                   mask_feature=(FeatureType.MASK, "IS_VALID"),
                                                   copy_features=[(FeatureType.MASK_TIMELESS, "LULC")],
                                                   resample_range=resampled_range)

    # Save original LULC reference map. Output data contain bands and LULC.
    save = SaveTask("./datasets",
                    features=[(FeatureType.DATA, "BANDS"), (FeatureType.MASK_TIMELESS, "LULC"), ],
                    overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    workflow_nodes = linearly_connect_tasks(load, filter_task, linear_interpolation, save)
    workflow = EOWorkflow(workflow_nodes=workflow_nodes)

    return workflow_nodes, workflow


##
def save_patch(workflow_nodes, workflow, patch):
    SAVE_DIR = os.path.join(ROOT_DIR, "datasets")
    execution_args = []
    execution_args.append(
        {
            workflow_nodes[0]: {"eopatch_folder": f"./{patch}"},
            workflow_nodes[-1]: {"eopatch_folder": f"./{patch}"},
        }
    )
    executor = EOExecutor(workflow, execution_args, save_logs=False)
    executor.run(1)
    os.chdir(SAVE_DIR)
    eopatch = EOPatch.load(os.path.join(os.getcwd(), f"{patch}"), lazy_loading=True)
    # Save data in png format
    cv2.imwrite(f"LULC_{patch}.png", eopatch.mask_timeless["LULC"])
    # Transform data from float ranging from 0~1 to uint8 ranging from 0~255
    # Every patch will keep 21 images in 2019. The first and last two images are excluded since linear interpolation may return NAN.
    for idx, data in enumerate(eopatch.data["BANDS"][2:-2]):
        # choosing NGB (Near infrared) channel instead of RGB
        image = (np.clip(data[..., [3, 1, 0]] * 3, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"image_{idx}_{patch}.png", image)
    os.remove(f"./{patch}/data/BANDS.npy")
    os.rmdir(f"./{patch}/data")
    os.remove(f"./{patch}/mask_timeless/LULC.npy")
    os.rmdir(f"./{patch}/mask_timeless")
    os.rmdir(f"./{patch}")
    os.chdir(ROOT_DIR)


##
ROOT_DIR = sys.path[-1]
os.chdir(ROOT_DIR)

subfolder_list_clean = get_patch_list()

##
workflow_nodes, workflow = linear_task()
for patch in tqdm(subfolder_list_clean):
    save_patch(workflow_nodes, workflow, patch)

