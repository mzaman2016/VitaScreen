import time
from pathlib import Path

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import PolynomialFeatures

from data.IGTD_Functions import (
    min_max_transform,
    select_features_by_variation,
    table_to_image,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    df = pd.read_csv(cfg.data.file_path)
    logger.info(f"Data shape: {df.shape}")

    if "feature_cols" in cfg.data:
        feature_cols = cfg.data.feature_cols
    else:
        feature_cols = [col for col in df.columns if col != cfg.data.target_col]
    logger.info(f"Number of features: {len(feature_cols)}")

    # Create polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(df[feature_cols])

    poly_feature_names = poly.get_feature_names_out(feature_cols)
    df_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
    logger.info(f"Number of polynomial features: {df_poly.shape[1]}")

    # Select features with large variations across samples
    X = df_poly[poly_feature_names]
    id = select_features_by_variation(
        X, variation_measure="var", num=len(poly_feature_names)
    )
    X = X.iloc[:, id]

    # Perform min-max transformation
    X_norm = min_max_transform(X.values)
    X_norm = pd.DataFrame(X_norm, columns=X.columns, index=X.index)

    # Transform tabular data to images
    Path(cfg.igtd.output_dir).mkdir(exist_ok=True)

    start_time = time.time()
    table_to_image(
        norm_d=X_norm,
        scale=[cfg.igtd.nrows, cfg.igtd.ncols],
        fea_dist_method=cfg.igtd.fea_dist_method,
        image_dist_method=cfg.igtd.image_dist_method,
        save_image_size=cfg.igtd.save_image_size,
        max_step=cfg.igtd.max_step,
        val_step=cfg.igtd.val_step,
        normDir=cfg.igtd.output_dir,
        error=cfg.igtd.error,
    )
    end_time = time.time()

    logger.info(
        f"Image transformation completed in {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    main()
