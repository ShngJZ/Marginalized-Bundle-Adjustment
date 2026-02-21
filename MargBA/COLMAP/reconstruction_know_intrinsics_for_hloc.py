"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Functions to use in HLOC, to obtain a COLMAP reconstruction, considering
known intrinsics"""

from typing import Optional, List, Dict, Any
import pycolmap
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / '../../third_party/Hierarchical-Localization'))

from hloc import logger
from hloc.triangulation import (import_features, estimation_and_geometric_verification, import_matches)
from hloc.reconstruction import (create_empty_db, get_image_ids, run_reconstruction, import_images)

def reconstruction_w_known_intrinsics(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    cam: Any,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)

    # Set up database with known intrinsics using pycolmap.Database
    # An initial value for the focal length is obtained from the EXIF data when importing the images.
    # It is then refined during the reconstruction. It is possible to disable the refinement of intrinsics
    # during BA (like in the triangulation), but you then need to first update the database
    # database.db with the know intrinsics. This needs to be done for each camera
    # (or a single one if shared intrinsics), after import_images but before run_reconstruction.
    with pycolmap.Database.open(database) as db:
        cam_id = db.write_camera(cam)
        for i, name in enumerate(image_list):
            db.write_image(pycolmap.Image(name=name, camera_id=cam_id, image_id=i+1))

    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options
    )
    if reconstruction is not None:
        logger.info(
            f"Reconstruction statistics:\n{reconstruction.summary()}"
            + f"\n\tnum_input_images = {len(image_ids)}"
        )
    return reconstruction

