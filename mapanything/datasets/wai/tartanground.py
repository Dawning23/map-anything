# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
TartanGround Dataset using fisheye camera data converted from cubemap.

Data layout (per trajectory):
    ROOT/<scene>/{Data_omni,Data_diff}/<traj>/
        image_fisheye_CAM_{A,B,C,D}/  -> {frame:06d}_fisheye_CAM_X.png
        depth_fisheye_CAM_{A,B,C,D}/  -> {frame:06d}_fisheye_CAM_X_depth.npy  (ray-depth)
        pose_fisheye_CAM_{A,B,C,D}.txt   -> (N, 7) [tx,ty,tz, qx,qy,qz,qw]
        intrinsics_fisheye_CAM_{A,B,C,D}.json  -> MEI/OMNI params
"""

import json
import logging
import os
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mapanything.datasets.base.base_dataset import BaseDataset, is_good_type, view_name
from mapanything.utils.geometry import depthmap_to_camera_coordinates

logger = logging.getLogger(__name__)

CAMERA_NAMES = ["CAM_A", "CAM_B", "CAM_C", "CAM_D"]
DATA_SUBDIRS = ["Data_omni", "Data_diff"]


def _mei_unproject_rays(fx, fy, cx, cy, k1, k2, p1, p2, xi, H, W, max_iters=20):
    """
    Full MEI/OMNI unprojection: compute unit ray directions for each pixel.

    The MEI model unprojects pixels through: remove intrinsics → invert radtan
    distortion → MEI sphere model → normalize to unit vectors.

    Intrinsics (fx, fy, cx, cy) may differ from the original calibration when
    the image has been cropped/resized, but distortion params (k1..p2, xi) are
    invariant because normalized coordinates mx = (u - cx) / fx are preserved.

    Args:
        fx, fy, cx, cy: intrinsics (possibly adjusted for crop/resize)
        k1, k2, p1, p2, xi: MEI distortion / mirror parameters
        H, W: image dimensions

    Returns:
        rays: (H, W, 3) float32 — unit ray directions per pixel
        valid_mask: (H, W) bool — pixels with valid MEI unprojection
    """
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    mx = (uu - cx) / fx
    my = (vv - cy) / fy

    xr, yr = mx.copy(), my.copy()
    for _ in range(max_iters):
        r2 = xr**2 + yr**2
        radial = 1.0 + k1 * r2 + k2 * r2**2
        dx = 2.0 * p1 * xr * yr + p2 * (r2 + 2.0 * xr**2)
        dy = p1 * (r2 + 2.0 * yr**2) + 2.0 * p2 * xr * yr
        safe = np.where(np.abs(radial) < 1e-12, 1.0, radial)
        xr = (mx - dx) / safe
        yr = (my - dy) / safe

    rho2 = xr**2 + yr**2
    disc = 1.0 + (1.0 - xi**2) * rho2
    valid_mask = disc >= 0
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    Z = (sqrt_d - xi * rho2) / (rho2 + 1.0)
    X = xr * (Z + xi)
    Y = yr * (Z + xi)

    rays = np.stack([X, Y, Z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    rays /= norms

    valid_mask = valid_mask & (~np.isnan(rays).any(axis=-1)) & (Z > -0.5)
    return rays.astype(np.float32), valid_mask


def get_absolute_pointmaps_and_rays_info(
    depthmap, camera_intrinsics, camera_pose, calib, **kw
):
    """
    MEI fisheye version: compute pointmaps and ray info using MEI unprojection.

    Unlike the pinhole version in geometry.py, this function:
    - Uses MEI model to get proper unit ray directions (not [x/fx, y/fy, 1])
    - Treats depthmap as ray-depth (radial distance), not z-depth
    - Computes pts_cam = unit_ray × ray_depth (radial distance parameterization)

    This is consistent with UniK3D's formulation:
        point = ray × distance, where ||ray|| = 1 and distance = ||point||.

    Args:
        depthmap: (H, W) ray-depth array (radial distance from camera center)
        camera_intrinsics: 3x3 matrix (adjusted for crop/resize)
        camera_pose: 4x4 cam2world matrix
        calib: dict with MEI parameters (k1, k2, p1, p2, xi)

    Returns:
        pts_world: (H, W, 3) world-frame 3D points
        valid_mask: (H, W) bool
        ray_origins_world: (H, W, 3)
        ray_directions_world: (H, W, 3) unit ray directions in world frame
        depth_along_ray: (H, W, 1) radial distance
        ray_directions_cam: (H, W, 3) unit ray directions in camera frame
        pts_cam: (H, W, 3) camera-frame 3D points
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    ray_directions_cam, mei_valid = _mei_unproject_rays(
        camera_intrinsics[0, 0], camera_intrinsics[1, 1],
        camera_intrinsics[0, 2], camera_intrinsics[1, 2],
        calib["k1"], calib["k2"], calib["p1"], calib["p2"],
        calib["xi"], H, W,
    )

    ray_depth = depthmap[..., None]  # (H, W, 1)
    pts_cam = (ray_directions_cam * ray_depth).astype(np.float32)
    depth_along_ray = ray_depth.astype(np.float32)

    valid_mask = (depthmap > 0.0) & mei_valid

    ray_origins_world = np.zeros_like(ray_directions_cam)
    ray_directions_world = ray_directions_cam.copy()
    pts_world = pts_cam.copy()
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]
        ray_origins_world = ray_origins_world + t_cam2world[None, None, :]
        ray_directions_world = np.einsum(
            "ik, vuk -> vui", R_cam2world, ray_directions_cam
        )
        pts_world = ray_origins_world + ray_directions_world * depth_along_ray

    return (
        pts_world,
        valid_mask,
        ray_origins_world,
        ray_directions_world,
        depth_along_ray,
        ray_directions_cam,
        pts_cam,
    )


class TartanGroundWAI(BaseDataset):
    """
    TartanGround dataset with synthetic fisheye camera views.

    Each scene contains multiple trajectories under Data_omni/, each trajectory
    having 4 fisheye cameras (CAM_A..D).  For each __getitem__ call, one
    trajectory and one frame are randomly selected, then `num_views_to_sample`
    cameras are drawn to form the multi-view set.
    """

    def __init__(
        self,
        *args,
        ROOT,
        split,
        dataset_metadata_dir=None,
        overfit_num_sets=None,
        debug_by_vis=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.split = split
        assert split in ["train", "val"], "split must be train or val"

        self.debug_by_vis = debug_by_vis
        self.is_metric_scale = True
        self.is_synthetic = True

        if split == "train":
            self.scenes = [
                "CarWelding", "Downtown", "AbandonedCable", "SeasonalForestAutumn",
                "OldTownFall", "Gascola", "SeasonalForestWinterNight", "Fantasy",
                "CastleFortress", "EndofTheWorld", "OldBrickHouseNight", "Slaughter",
                "OldIndustrialCity", "OldTownWinter", "AbandonedFactory", "JapaneseAlley",
                "ForestEnv", "SeasonalForestWinter", "HQWesternSaloon", "Ruins",
                "CoalMine", "Office", "ModernCityDowntown", "ModularNeighborhood",
                "WesternDesertTown", "OldScandinavia", "ModUrbanCity", "JapaneseCity",
                "ModularNeighborhoodIntExt", "Prison", "GreatMarsh", "AbandonedSchool",
                "HongKong", "FactoryWeather", "UrbanConstruction", "MiddleEast",
                "DesertGasStation", "AmusementPark", "Antiquity3D", "AbandonedFactory2",
                "OldTownSummer", "OldTownNight", "WaterMillDay", "NordicHarbor",
                "House", "CyberPunkDowntown", "GothicIsland", "SoulCity",
                "Sewerage", "Rome", "SeasonalForestSummerNight", "IndustrialHangar",
                "OldBrickHouseDay",
            ]
        else:
            self.scenes = [
                "VictorianStreet", "SeasideTown", "SeasonalForestSpring", "ConstructionSite",
                "BrushifyMoon", "Restaurant", "Hospital", "WaterMillNight",
                "AncientTowns", "Supermarket",
            ]

        # Discover available trajectories per scene (from Data_omni and Data_diff)
        self._scene_trajs = {}
        for scene in self.scenes:
            entries = []
            for subdir in DATA_SUBDIRS:
                data_dir = os.path.join(ROOT, scene, subdir)
                if not os.path.isdir(data_dir):
                    continue
                for d in sorted(os.listdir(data_dir)):
                    if os.path.isdir(os.path.join(data_dir, d)):
                        entries.append((subdir, d))
            if entries:
                self._scene_trajs[scene] = entries

        self.scenes = [s for s in self.scenes if s in self._scene_trajs]
        if overfit_num_sets is not None:
            self.scenes = self.scenes[:overfit_num_sets]
        self.num_of_scenes = len(self.scenes)

        # Load sky class label per scene from seg_label_map.json
        self._sky_labels = {}
        for scene in self.scenes:
            label_map_path = os.path.join(ROOT, scene, "seg_label_map.json")
            if not os.path.isfile(label_map_path):
                continue
            with open(label_map_path) as f:
                label_map = json.load(f)
            sky_val = label_map.get("name_map", {}).get("sky")
            if sky_val is not None:
                self._sky_labels[scene] = int(sky_val)
        logger.info("Loaded sky labels for %d / %d scenes", len(self._sky_labels), len(self.scenes))

        # Pre-compute MEI valid masks (same calibration across all scenes)
        self._fisheye_valid = {}
        self._load_camera_luts()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_camera_luts(self):
        """Load MEI intrinsics from the first available trajectory and pre-compute valid masks."""
        for scene in self.scenes:
            first_subdir, first_traj = self._scene_trajs[scene][0]
            traj_path = os.path.join(self.ROOT, scene, first_subdir, first_traj)

            loaded_all = True
            for cam_name in CAMERA_NAMES:
                path = os.path.join(traj_path, f"intrinsics_fisheye_{cam_name}.json")
                if not os.path.exists(path):
                    loaded_all = False
                    break
                with open(path) as f:
                    calib = json.load(f)
                W_orig, H_orig = int(calib["width"]), int(calib["height"])
                _, valid = _mei_unproject_rays(
                    calib["fx"], calib["fy"], calib["cx"], calib["cy"],
                    calib["k1"], calib["k2"], calib["p1"], calib["p2"],
                    calib["xi"], H_orig, W_orig,
                )
                self._fisheye_valid[cam_name] = valid

            if loaded_all:
                return

        logger.warning("Could not load MEI intrinsics — valid mask will be unavailable.")

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]

        # Pick a random trajectory within the scene
        trajs = self._scene_trajs[scene_name]
        traj_idx = int(self._rng.integers(0, len(trajs)))
        data_subdir, traj_name = trajs[traj_idx]
        traj_path = os.path.join(self.ROOT, scene_name, data_subdir, traj_name)

        # Count available frames (use first camera as reference)
        sample_dir = os.path.join(traj_path, f"image_fisheye_{CAMERA_NAMES[0]}")
        frame_files = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
        num_frames = len(frame_files)
        assert num_frames > 0, f"No frames found in {sample_dir}"

        # Pick a random frame
        frame_idx = int(self._rng.integers(0, num_frames))

        views = []
        for ci in range(self.num_views):
            cam_name = CAMERA_NAMES[ci]

            # --- Image ---
            img_path = os.path.join(
                traj_path, f"image_fisheye_{cam_name}",
                f"{frame_idx:06d}_fisheye_{cam_name}.png",
            )
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --- Ray-depth (radial distance from camera center) ---
            depth_path = os.path.join(
                traj_path, f"depth_fisheye_{cam_name}",
                f"{frame_idx:06d}_fisheye_{cam_name}_depth.npy",
            )
            assert os.path.exists(depth_path), f"Depth not found: {depth_path}"
            ray_depth = np.load(depth_path).astype(np.float32)

            if cam_name in self._fisheye_valid:
                fisheye_valid = self._fisheye_valid[cam_name]
                if fisheye_valid.shape == ray_depth.shape:
                    ray_depth[~fisheye_valid] = 0.0

            depthmap = np.nan_to_num(ray_depth, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Intrinsics (MEI fx/fy/cx/cy as pinhole approximation) ---
            intrinsics_path = os.path.join(
                traj_path, f"intrinsics_fisheye_{cam_name}.json",
            )
            with open(intrinsics_path) as f:
                calib = json.load(f)
            intrinsics = np.float32([
                [calib["fx"], 0, calib["cx"]],
                [0, calib["fy"], calib["cy"]],
                [0, 0, 1],
            ])

            # --- Pose (c2w) ---
            pose_path = os.path.join(traj_path, f"pose_fisheye_{cam_name}.txt")
            poses_all = np.loadtxt(pose_path)
            t = poses_all[frame_idx, :3]
            q = poses_all[frame_idx, 3:7]  # [qx, qy, qz, qw]
            R = Rotation.from_quat(q).as_matrix()
            c2w_pose = np.eye(4, dtype=np.float32)
            c2w_pose[:3, :3] = R.astype(np.float32)
            c2w_pose[:3, 3] = t.astype(np.float32)

            # --- Fisheye segmentation → non_ambiguous_mask (exclude sky) ---
            non_ambiguous_mask = None
            if scene_name in self._sky_labels:
                seg_path = os.path.join(
                    traj_path, f"seg_fisheye_{cam_name}",
                    f"{frame_idx:06d}_fisheye_{cam_name}_seg.png",
                )
                if os.path.exists(seg_path):
                    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                    if seg_img is not None:
                        sky_val = self._sky_labels[scene_name]
                        non_ambiguous_mask = (seg_img != sky_val).astype(np.float32)

            # --- Crop / resize ---
            if non_ambiguous_mask is not None:
                image, depthmap, intrinsics, (non_ambiguous_mask,) = (
                    self._crop_resize_if_necessary(
                        image=image,
                        resolution=resolution,
                        depthmap=depthmap,
                        intrinsics=intrinsics,
                        additional_quantities=[non_ambiguous_mask],
                    )
                )
            else:
                image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                )

            view = dict(
                img=image,
                depthmap=depthmap,
                camera_pose=c2w_pose,
                camera_intrinsics=intrinsics,
                calib=calib,
                dataset="TartanGround",
                label=scene_name,
                instance=f"fisheye_{cam_name}/{frame_idx:06d}",
            )
            if non_ambiguous_mask is not None:
                view["non_ambiguous_mask"] = non_ambiguous_mask
            views.append(view)

        return views


    def _getitem_fn(self, idx):
        if isinstance(idx, tuple):
            # The idx is a tuple if specifying the aspect-ratio or/and the number of views
            if isinstance(self.num_views, int):
                idx, ar_idx = idx
            else:
                idx, ar_idx, num_views_to_sample_idx = idx
        else:
            assert len(self._resolutions) == 1
            assert isinstance(self.num_views, int)
            ar_idx = 0

        # Setup the rng
        if self.seed:  # reseed for each _getitem_fn
            # Leads to deterministic sampling where repeating self.seed and self._seed_offset yields the same multi-view set again
            # Scenes will be repeated if size of dataset is artificially increased using "N @" or "N *"
            # When scenes are repeated, self._seed_offset is increased to ensure new multi-view sets
            # This is useful for evaluation if the number of dataset scenes is < N, yet we want unique multi-view sets each iter
            self._rng = np.random.default_rng(seed=self.seed + self._seed_offset + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # Get the views for the given index and check that the number of views is correct
        resolution = self._resolutions[ar_idx]
        if isinstance(self.num_views, int):
            num_views_to_sample = self.num_views
        else:
            num_views_to_sample = self.num_views[num_views_to_sample_idx]
        views = self._get_views(idx, num_views_to_sample, resolution)
        if isinstance(self.num_views, int):
            assert len(views) == self.num_views
        else:
            assert len(views) in self.num_views

        for v, view in enumerate(views):
            # Store the index and other metadata
            view["idx"] = (idx, ar_idx, v)
            view["is_metric_scale"] = self.is_metric_scale
            view["is_synthetic"] = self.is_synthetic

            # Check the depth, intrinsics, and pose data (also other data if present)
            assert "camera_intrinsics" in view
            assert "camera_pose" in view
            assert np.isfinite(view["camera_pose"]).all(), (
                f"NaN or infinite values in camera pose for view {view_name(view)}"
            )
            assert np.isfinite(view["depthmap"]).all(), (
                f"NaN or infinite values in depthmap for view {view_name(view)}"
            )
            assert "valid_mask" not in view
            assert "pts3d" not in view, (
                f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            )
            if "prior_depth_z" in view:
                assert np.isfinite(view["prior_depth_z"]).all(), (
                    f"NaN or infinite values in prior_depth_z for view {view_name(view)}"
                )
            if "non_ambiguous_mask" in view:
                assert np.isfinite(view["non_ambiguous_mask"]).all(), (
                    f"NaN or infinite values in non_ambiguous_mask for view {view_name(view)}"
                )

            # Encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            if self.debug_by_vis:
                view["_debug_rgb"] = np.asarray(view["img"]).copy()
            view["img"] = self.transform(view["img"])
            view["data_norm_type"] = self.data_norm_type

            # Compute the pointmaps, raymap and depth along ray
            (
                pts3d,
                valid_mask,
                ray_origins_world,
                ray_directions_world,
                depth_along_ray,
                ray_directions_cam,
                pts3d_cam,
            ) = get_absolute_pointmaps_and_rays_info(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            depth_flat = view["depthmap"][view["valid_mask"]]
            if depth_flat.size > 0:
                depth_95 = min(np.percentile(depth_flat, 95), 200.0)
            else:
                depth_95 = 200.0
            view["valid_mask"] &= view["depthmap"] <= depth_95
            view["depth_along_ray"] = depth_along_ray
            view["ray_directions_cam"] = ray_directions_cam
            view["pts3d_cam"] = pts3d_cam

            # Compute the prior depth along ray if present
            if "prior_depth_z" in view:
                prior_pts3d, _ = depthmap_to_camera_coordinates(
                    view["prior_depth_z"], view["camera_intrinsics"]
                )
                view["prior_depth_along_ray"] = np.linalg.norm(prior_pts3d, axis=-1)
                view["prior_depth_along_ray"] = view["prior_depth_along_ray"][..., None]
                del view["prior_depth_z"]

            # Convert ambiguous mask dtype to match valid mask dtype
            if "non_ambiguous_mask" in view:
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )
            else:
                ambiguous_mask = view["depthmap"] < 0
                view["non_ambiguous_mask"] = ~ambiguous_mask
                view["non_ambiguous_mask"] = view["non_ambiguous_mask"].astype(
                    view["valid_mask"].dtype
                )

            # Check all datatypes
            view.pop("calib", None)
            for key, val in view.items():
                res, err_msg = is_good_type(val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            # Check shapes
            assert view["depthmap"].shape == view["img"].shape[1:]
            assert view["depthmap"].shape == view["pts3d"].shape[:2]
            assert view["depthmap"].shape == view["valid_mask"].shape
            assert view["depthmap"].shape == view["depth_along_ray"].shape[:2]
            assert view["depthmap"].shape == view["ray_directions_cam"].shape[:2]
            assert view["depthmap"].shape == view["pts3d_cam"].shape[:2]
            if "prior_depth_along_ray" in view:
                assert view["depthmap"].shape == view["prior_depth_along_ray"].shape[:2]
            if "non_ambiguous_mask" in view:
                assert view["depthmap"].shape == view["non_ambiguous_mask"].shape

            # Expand the last dimension of the depthmap
            view["depthmap"] = view["depthmap"][..., None]

            # Append RNG state to the views, this allows to check whether the RNG is in the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            # Compute and store the quaternions and translation for the camera poses
            # Notation is (x, y, z, w) for quaternions
            # This also ensures that the camera poses have a positive determinant (right-handed coordinate system)
            view["camera_pose_quats"] = (
                Rotation.from_matrix(view["camera_pose"][:3, :3])
                .as_quat()
                .astype(view["camera_pose"].dtype)
            )
            view["camera_pose_trans"] = view["camera_pose"][:3, 3].astype(
                view["camera_pose"].dtype
            )

            # Check the pointmaps, rays, depth along ray, and camera pose quaternions and translation to ensure they are finite
            assert np.isfinite(view["pts3d"]).all(), (
                f"NaN in pts3d for view {view_name(view)}"
            )
            assert np.isfinite(view["valid_mask"]).all(), (
                f"NaN in valid_mask for view {view_name(view)}"
            )
            assert np.isfinite(view["depth_along_ray"]).all(), (
                f"NaN in depth_along_ray for view {view_name(view)}"
            )
            assert np.isfinite(view["ray_directions_cam"]).all(), (
                f"NaN in ray_directions_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["pts3d_cam"]).all(), (
                f"NaN in pts3d_cam for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_quats"]).all(), (
                f"NaN in camera_pose_quats for view {view_name(view)}"
            )
            assert np.isfinite(view["camera_pose_trans"]).all(), (
                f"NaN in camera_pose_trans for view {view_name(view)}"
            )
            if "prior_depth_along_ray" in view:
                assert np.isfinite(view["prior_depth_along_ray"]).all(), (
                    f"NaN in prior_depth_along_ray for view {view_name(view)}"
                )
        if self.debug_by_vis:
            all_pts = []
            all_colors = []
            all_pts_cam = []
            all_quats = []
            all_trans = []
            for view in views:
                mask = view["valid_mask"]
                pts = view["pts3d"][mask]
                colors = view["_debug_rgb"][mask]
                all_pts.append(pts.reshape(-1, 3).astype(np.float32))
                all_colors.append(colors.reshape(-1, 3).astype(np.uint8))

                pts_cam = view["pts3d_cam"][mask]
                all_pts_cam.append(pts_cam.reshape(-1, 3).astype(np.float32))
                all_quats.append(view["camera_pose_quats"])
                all_trans.append(view["camera_pose_trans"])
                del view["_debug_rgb"]

            all_pts = np.concatenate(all_pts, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            N = len(all_pts)

            debug_dir = "/drobotics-ailab/bohao.zhang/Projects/map-anything/debug_by_vis"
            os.makedirs(debug_dir, exist_ok=True)

            def _save_ply(path, points, colors):
                n = len(points)
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {n}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n"
                )
                dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ])
                verts = np.empty(n, dtype=dtype)
                verts['x'] = points[:, 0]
                verts['y'] = points[:, 1]
                verts['z'] = points[:, 2]
                verts['r'] = colors[:, 0]
                verts['g'] = colors[:, 1]
                verts['b'] = colors[:, 2]
                with open(path, 'wb') as f:
                    f.write(header.encode('ascii'))
                    f.write(verts.tobytes())

            ply_path = os.path.join(debug_dir, f"debug_pts3d_idx{idx}_{views[0]['label']}.ply")
            _save_ply(ply_path, all_pts, all_colors)
            logger.info(f"Saved debug PLY ({N} points) to {ply_path}")

            # Transform pts3d_cam to world coordinates using camera_pose_quats and camera_pose_trans
            all_pts_cam_to_world = []
            for pts_cam_i, quat_i, trans_i in zip(all_pts_cam, all_quats, all_trans):
                R_c2w = Rotation.from_quat(quat_i).as_matrix().astype(np.float32)
                pts_world_i = (pts_cam_i @ R_c2w.T) + trans_i[None, :]
                all_pts_cam_to_world.append(pts_world_i)
            all_pts_cam_to_world = np.concatenate(all_pts_cam_to_world, axis=0)

            ply_path_cam2world = os.path.join(
                debug_dir, f"debug_pts3d_cam2world_idx{idx}_{views[0]['label']}.ply"
            )
            _save_ply(ply_path_cam2world, all_pts_cam_to_world, all_colors)
            logger.info(
                f"Saved cam2world debug PLY ({len(all_pts_cam_to_world)} points) to {ply_path_cam2world}"
            )

        return views

# ======================================================================
# Standalone test / visualization
# ======================================================================


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/drobotics-ailab/bohao.zhang/Projects/map-anything/tartanground", type=str
    )
    parser.add_argument("--viz", action="store_true", default=True)

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = TartanGroundWAI(
        num_views=4,
        split="train",
        covisibility_thres=0.25,
        ROOT=args.root_dir,
        resolution=(518, 322),
        aug_crop=0,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
        max_num_retries=10,
        debug_by_vis=True,
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "TartanGround_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=10, replace=True)

    for num, idx in enumerate(tqdm(sampled_indices)):
        views = dataset[idx]