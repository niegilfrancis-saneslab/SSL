import typing as tp
from functools import partial

import numpy as np
import pandas as pd


def vector_interpolate(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    left: float | None = None,
    right: float | None = None,
) -> np.ndarray:
    """Linear interpolation for vectors.
    Args:
        x (np.ndarray): Points to sample. (N,)
        xp (np.ndarray): Domain of f. (P,)
        fp (np.ndarray): Values of f at `xp`. (P, D)
    Returns:
        np.ndarray: Interpolated values from f at `x`.
    """
    i = np.searchsorted(xp, x)
    if i == 0:
        return np.full_like(fp[0], left, dtype=fp.dtype) if left is not None else fp[0]
    if i == len(xp):
        return (
            np.full_like(fp[0], right, dtype=fp.dtype) if right is not None else fp[-1]
        )

    a, b = xp[i - 1], xp[i]
    fa, fb = fp[i - 1], fp[i]
    return fa + (fb - fa) * (x - a) / (b - a)


def validate_points(
    nose_p: np.ndarray,
    head_p: np.ndarray,
    ear_l_p: np.ndarray,
    ear_r_p: np.ndarray,
    neck_p: np.ndarray,
    body_1_p: np.ndarray,
    body_2_p: np.ndarray,
    body_3_p: np.ndarray,
    arena_corner_points: np.ndarray,
):
    """Using the available track nodes, computes the best estimate
    of animal head position and direction for dataset generation.

    Missing points should be indicated by NaN entries

    Cases:
        - Case 1: Head, ear_l, and ear_r present: use head for position, triangulate direction from ear_l and ear_r
        - Case 2: Head, neck, and one ear present: use body 1, 2, and 3 for direction (possible for ears to be swapped)
        - Case 3: ear_l, ear_r present, no head or neck: use body 1 and 2 to find nose.
        - Case 4: Only head and nose present: use head and nose
        - Case 5: Only body points present: discard frame
        - Case 6: Head and neck present: extrapolate to nose
        - Case 7: Head/nose/body points out of arena bounds: use body points to get direction, project to quad to get position

    Args:
        nose_p (np.ndarray): Nose point (N, 2,)
        head_p (np.ndarray): Head point (N, 2,)
        ear_l_p (np.ndarray): Left ear point (N, 2,)
        ear_r_p (np.ndarray): Right ear point (N, 2,)
        neck_p (np.ndarray): Neck point (N, 2,)
        body_1_p (np.ndarray): First body point (nearest to neck) (N, 2,)
        body_2_p (np.ndarray): Second body point (N, 2,)
        body_3_p (np.ndarray): Third body point (furthest to neck) (N, 2,)
        arena_corner_points (np.ndarray): Arena corner points (4, 2)
    """

    have: dict[str, np.ndarray] = {
        "nose": ~np.isnan(nose_p).any(axis=1),
        "head": ~np.isnan(head_p).any(axis=1),
        "ear_l": ~np.isnan(ear_l_p).any(axis=1),
        "ear_r": ~np.isnan(ear_r_p).any(axis=1),
        "neck": ~np.isnan(neck_p).any(axis=1),
        "body_1": ~np.isnan(body_1_p).any(axis=1),
        "body_2": ~np.isnan(body_2_p).any(axis=1),
        "body_3": ~np.isnan(body_3_p).any(axis=1),
    }

    for i in range(len(nose_p)):
        if have["head"][i] and have["ear_l"][i] and have["ear_r"][i]:
            # Case 1
            pass
        elif (
            have["head"][i]
            and have["neck"][i]
            and (have["ear_l"][i] or have["ear_r"][i])
        ):
            # case 2
            pass
        elif (
            have["ear_l"][i]
            and have["ear_r"][i]
            and have["body_1"][i]
            and have["body_2"][i]
        ):
            # case 3
            pass
        elif (
            have["head"][i]
            and have["nose"][i]
            and not (have["ear_l"][i] or have["ear_r"][i])
        ):
            # case 4
            pass
        elif have["head"][i] and have["neck"][i] and not have["nose"][i]:
            # case 6
            pass
        elif (
            have["body_1"][i]
            and have["body_2"][i]
            and have["body_3"][i]
            and not any(
                [
                    have["nose"][i] and point_in_quad(nose_p[i], arena_corner_points),
                    have["head"][i] and point_in_quad(head_p[i], arena_corner_points),
                    have["ear_l"][i] and point_in_quad(ear_l_p[i], arena_corner_points),
                    have["ear_r"][i] and point_in_quad(ear_r_p[i], arena_corner_points),
                    have["neck"][i] and point_in_quad(neck_p[i], arena_corner_points),
                ]
            )
        ):
            # case 7
            pass
        else:
            # case 5: discard frame
            pass

def find_id(
    t: np.ndarray,
    t_p: np.ndarray,
    track_ids,) :
    """
    Function to return the animal ID string corresponding to the time 't' provided
    
    t (np.ndarray): Times where we want the nose position
    t_p (np.ndarray): Times where the animal pose (any node) is available
    ids (np.ndarray): track id string

    Returns: 
        np.ndarray: Animal ID string nose coordinates at times `t`
    """

    id = np.full((len(t), 1), np.nan, dtype="S40")
    for i, t_i in enumerate(t):
        # Finds j such that t_p[j-1] < t_i <= t_p[j]
        j = np.searchsorted(t_p, t_i)
        
        id[i] = track_ids[j-1]

    return id



def nose_interpolator(
    t: np.ndarray,
    t_p: np.ndarray,
    nose_p: np.ndarray,
    head_p: np.ndarray,
    ear_l_p: np.ndarray,
    ear_r_p: np.ndarray,
) -> np.ndarray:
    """Custom interpolation function for the gerbil's nose node at time `t`

    Args:
        t (np.ndarray): Times where we want the nose position
        t_p (np.ndarray): Times where the animal pose (any node) is available
        nose_p (np.ndarray): Nose coords at times `t_p`. May include nan
        head_p (np.ndarray): Head coords at times `t_p`. May include nan
        ear_l_p (np.ndarray): Left ear coord at times `t_p`. May include nan
        ear_r_p (np.ndarray): Right ear coord at times `t_p`. May include nan

    Returns:
        np.ndarray: Interpolated nose coordinates at times `t`
    """
    nose_t = np.full((len(t), 2), np.nan, dtype=np.float32)
    # Determine the distance between the nose and the head
    nose_n_head_mask = ((~np.isnan(nose_p)) & (~np.isnan(head_p))).all(axis=-1)
    standard_nose_head_dist = np.median(
        np.linalg.norm(
            head_p[nose_n_head_mask, :] - nose_p[nose_n_head_mask, :],
            axis=-1,
        ),
    )

    # We gotta do this the slow way
    for i, t_i in enumerate(t):
        # Finds j such that t_p[j-1] < t_i <= t_p[j]
        j = np.searchsorted(t_p, t_i)
        # Determine if we can use normal interpolation or if special reconstruction is needed
        should_use_ears = (
            j <= 0
            or j >= len(t_p)
            or np.isnan(nose_p[j - 1]).any()
            or np.isnan(nose_p[j]).any()
        )
        if not should_use_ears:
            # normal interpolation
            nose_t[i] = vector_interpolate(
                t_i,
                t_p,
                nose_p,
            )
            continue

        # Reconstruct the nose position using the head and ears
        if (
            j >= len(head_p)
            or np.isnan(head_p[j - 1]).any()
            or np.isnan(head_p[j]).any()
        ):
            # If head is not available, we can't reconstruct
            # print(f"Cannot reconstruct nose at {t_i} because head is not available")
            continue

        head_p_j = vector_interpolate(t_i, t_p, head_p)

        # Use the ear position to find the direction to the nose from the head
        # Since the y axis is flipped, this is a clockwise rotation
        rot90_matrix = np.array([[0, -1], [1, 0]])
        if not np.isnan(ear_l_p[j - 1]).any() and not np.isnan(ear_l_p[j]).any():
            # Use left ear
            ear_l_j = vector_interpolate(
                t_i,
                t_p,
                ear_l_p,
            )
            head_to_nose = rot90_matrix @ (ear_l_j - head_p_j)
        elif not np.isnan(ear_r_p[j - 1]).any() and not np.isnan(ear_r_p[j]).any():
            # Use right ear
            ear_r_j = vector_interpolate(
                t_i,
                t_p,
                ear_r_p,
            )
            head_to_nose = rot90_matrix @ (head_p_j - ear_r_j)
        else:
            # If neither ear is available, we can't reconstruct
            # print(f"Cannot reconstruct nose at {t_i} because no ears are available")
            continue

        head_to_nose /= np.linalg.norm(head_to_nose)
        head_to_nose *= standard_nose_head_dist
        nose_t[i] = head_p_j + head_to_nose

    return nose_t


class Animal:
    low_score_threshold = 0.2

    def __init__(
        self,
        frame_indices: np.ndarray,
        nodes: dict[str, np.ndarray],
        node_scores: dict[str, np.ndarray],
        track_id,
        node_names: tp.Optional[list[str]] = None,
    ):
        """A wrapper around animal track data which allows for automatic interpolation.

        Args:
            frame_indices (np.ndarray): Indices (in the video) associated with each track.
                Should have same length as all value arrays in nodes
            nodes (dict[str, np.ndarray]): Dictionary of (n, 2) arrays containing (x,y) coordinates for each track
            node_scores (dict[str, np.ndarray]): Dictionary of (n,) arrays containing SLEAP scores for each track
            node_names (tp.Optional[list[str]]): Ordering to use for nodes.
        """

        self.node_names = node_names
        if node_names is None:
            self.node_names = list(nodes.keys())
        self.nodes = nodes
        self.scores = node_scores
        self.indices = frame_indices
        self.models: dict[str, "partial"] = {}
        self.track_id = track_id

        # Construct interpolation functions
        for dim_idx in range(2):
            head_1d = self.nodes["head"][:, dim_idx]  # 1d
            # For each node, create a subset of frame_idx where no nans are present
            valid_mask = ~np.isnan(head_1d)
            valid_mask &= ~np.isnan(self.scores["head"])
            # ---------------------------------Removing the low score thresholding for head point---------------------------------
            #weight = self.scores["head"]
            #valid_mask &= weight > Animal.low_score_threshold

            head_1d = head_1d[valid_mask]
            idx = self.indices[valid_mask]  # 1d
            # Threshold for low scores
            model = partial(np.interp, xp=idx, fp=head_1d, left=np.nan, right=np.nan)
            dim = ("x", "y")[dim_idx]
            self.models[f"head.{dim}"] = model
        # Nose interpolation is more complex, so we use a custom function
        nose = self.nodes["nose"]
        head = self.nodes["head"]
        ear_l = self.nodes["ear_l"]
        ear_r = self.nodes["ear_r"]
        self.models["nose"] = partial(
            nose_interpolator,
            t_p=self.indices,
            nose_p=nose,
            head_p=head,
            ear_l_p=ear_l,
            ear_r_p=ear_r,
        )

        self.models["track_id"] = partial(find_id,
                                          t_p=self.indices,
                                          track_ids = self.track_id
                                          )

    def __getitem__(self, frame_idx: int | np.ndarray | tuple):
        should_squeeze = False
        arr_slice = (slice(None, None, None),)
        arr_slice_id = (slice(None,None),)
        # tuple unpacking
        if isinstance(frame_idx, tuple):
            arr_slice = (slice(None, None, None), *frame_idx[1:])
            arr_slice_id = (slice(None, None), *frame_idx[1:])
            frame_idx = frame_idx[0]

        if isinstance(frame_idx, int):
            frame_idx = np.array([frame_idx])
            should_squeeze = True
        elif isinstance(frame_idx, np.ndarray):
            pass
        elif isinstance(frame_idx, slice):
            start = frame_idx.start or 0
            stop = frame_idx.stop or self.indices.max() + 1
            step = frame_idx.step or 1
            frame_idx = np.arange(start, stop, step)
        else:
            raise ValueError(
                f"Invalid indexing argument type: {type(frame_idx)}. Expected int, np.ndarray, or slice."
            )

        nose_xy = self.models["nose"](frame_idx)
        head_xy = np.stack(
            [
                self.models["head.x"](frame_idx),
                self.models["head.y"](frame_idx),
            ],
            axis=-1,
        )  # Both are (batch, 2)
        tracks = np.stack([nose_xy, head_xy], axis=-2)
        ids = self.models["track_id"](frame_idx)

        if should_squeeze:
            tracks = np.squeeze(tracks, axis=0)
            ids = np.squeeze(ids, axis=0)

        return tracks[arr_slice],ids  # (batch, nodes, 2), (batch, 1)

    def __setitem__(self, *args):
        raise AttributeError("Animal is immutable")

    def __delitem__(self, *args):
        raise AttributeError("Animal is immutable")

    def __len__(self):
        return self.indices.max()


def extract_tracks(df: pd.DataFrame) -> list[Animal]:
    """Extracts animal tracks from a DataFrame containing SLEAP tracking data.

    Args:
        df (pd.DataFrame): DataFrame containing SLEAP tracking data with columns for animal tracks.
        This script uses the nose and head nodes, so the columns should include nose.x, nose.y, nose.score,
        and the corresponding columns for head.

    Returns:
        list[Animal]: A list of Animal instances, each representing a tracked animal with interpolated coordinates.
    """
    # For each animal and node of interest, extract the coords, frame indices, and scores:
    # Currently care about nose and head
    nodes_we_care_abt = ("nose", "head", "ear_l", "ear_r")
    animal_ids = df["track"].unique()
    animal_tracks = []
    for animal in animal_ids:
        if "none" in animal:
            continue 
        animal_df = df[df["track"] == animal]
        frame_indices = animal_df["frame_idx"].to_numpy()
        track_ids = np.array(animal_df[df["track"] == animal]["track"].to_list(),dtype="S40")
        nodes = {}
        scores = {}
        for node_name in nodes_we_care_abt:
            x = animal_df[f"{node_name}.x"].to_numpy()
            y = animal_df[f"{node_name}.y"].to_numpy()
            score = animal_df[f"{node_name}.score"].to_numpy()
            nodes[node_name] = np.stack([x, y], axis=-1)
            scores[node_name] = score
        animal_instance = Animal(frame_indices, nodes, scores, track_ids, nodes_we_care_abt)
        animal_tracks.append(animal_instance)
    return animal_tracks


def triangle_area(points: np.ndarray) -> float:
    """Returns the area of the triangle defined by the three points

    Args:
        points: A 3x2 numpy array of the three points
    """
    a, b, c = points
    return 0.5 * np.abs(np.cross(b - a, c - a))


def project_point_to_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Projects a point onto a line defined by two points

    Args:
        point: A 2D numpy array of the point to project
        line: A 2x2 numpy array of the two points defining the line
    """

    # vec from a to b
    a, b = line
    p = point
    ab = b - a
    ap = p - a
    return a + np.dot(ap, ab) / np.dot(ab, ab) * ab


def point_in_quad(point: np.ndarray, corner_points: np.ndarray) -> bool:
    """Determines if a point is inside the quadrilateral defined by corner_points

    Args:
        point: A 2D numpy array of the point to check
        corner_points: A 4x2 numpy array of the four corner points defining the quadrilateral
    """
    tri_areas = (
        triangle_area([point, *corner_points[(0, 1), :]]),
        triangle_area([point, *corner_points[(1, 2), :]]),
        triangle_area([point, *corner_points[(2, 3), :]]),
        triangle_area([point, *corner_points[(3, 0), :]]),
    )
    quad_area = triangle_area(corner_points[(0, 1, 2), :]) + triangle_area(
        corner_points[(0, 2, 3), :]
    )
    return np.sum(tri_areas) <= quad_area


def project_to_bounds(tracks: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Gets the projection of tracks onto the quadrilateral defined by corner_points
    This removes changes in the observed x and y coordinates that are due to climbing
    along the border wall.
    """

    centroid = corner_points.mean(axis=0)

    orig_shape = tracks.shape
    tracks = tracks.reshape(-1, 2)

    # assumes the points are ordered clockwise or counterclockwise
    quad_area = triangle_area(corner_points[(0, 1, 2), :]) + triangle_area(
        corner_points[(0, 2, 3), :]
    )

    # Approx len of the 0-1 and 2-3 edges
    edge_1_len = min(
        np.linalg.norm(corner_points[0] - corner_points[1]),
        np.linalg.norm(corner_points[3] - corner_points[2]),
    )
    # Approx len of the 1-2 and 3-0 edges
    edge_2_len = min(
        np.linalg.norm(corner_points[1] - corner_points[2]),
        np.linalg.norm(corner_points[0] - corner_points[3]),
    )

    proj_tracks = []
    for t in tracks:
        # Determine if the point is outside the quadrilateral
        # If it is, project it onto the nearest edge
        tri_areas = (
            triangle_area([t, *corner_points[(0, 1), :]]),
            triangle_area([t, *corner_points[(1, 2), :]]),
            triangle_area([t, *corner_points[(2, 3), :]]),
            triangle_area([t, *corner_points[(3, 0), :]]),
        )

        in_quad = np.sum(tri_areas) <= quad_area
        if in_quad:
            proj_tracks.append(t)
            continue

        # Project the point onto the nearest edge
        edge_dists = [
            np.linalg.norm(np.cross(t - corner_points[i], t - corner_points[j]))
            / np.linalg.norm(corner_points[i] - corner_points[j])
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]
        ]
        if edge_dists[0] > edge_2_len:
            # The point is far from the 0-1 edge, project to the 2-3 line
            t = project_point_to_line(t, corner_points[(2, 3), :])
        if edge_dists[1] > edge_1_len:
            # The point is far from the 1-2 edge, project to the 3-0 line
            t = project_point_to_line(t, corner_points[(3, 0), :])
        if edge_dists[2] > edge_2_len:
            # The point is far from the 2-3 edge, project to the 0-1 line
            t = project_point_to_line(t, corner_points[(0, 1), :])
        if edge_dists[3] > edge_1_len:
            # The point is far from the 3-0 edge, project to the 1-2 line
            t = project_point_to_line(t, corner_points[(1, 2), :])

        proj_tracks.append(t)
    proj_tracks = np.array(proj_tracks)
    proj_tracks = proj_tracks.reshape(orig_shape)

    return proj_tracks


if __name__ == "__main__":
    test_file = "/home/atanelus/box_data/experiment_135/idx_0/center-session_135_video-0.tracks.csv"
    df = pd.read_csv(test_file)
    print(df.columns)
    print(df.head())
    print()

    animals = extract_tracks(df)
    a0 = animals[0]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    nose = a0[:, 0]
    head = a0[:, 1]
    for i, animal in enumerate(animals):
        print(f"Animal {i}:")
        print(np.nanmin(animal[:, 1], axis=0))
        print(np.nanmax(animal[:, 1], axis=0))
        print(
            f"Proportion of nans in head: {np.isnan(animal[:, 1]).any(axis=-1).mean():.1%}"
        )
        print()

    print(a0.nodes["nose"].shape)
    ax.plot(nose[:, 0], nose[:, 1], label="nose, interpolated", c="blue", lw=0.5)
    ax.scatter(
        a0.nodes["nose"][:, 0],
        a0.nodes["nose"][:, 1],
        label="nose, original",
        s=4,
        c="red",
    )
    ax.legend()

    fig.tight_layout()
    plt.savefig("sample_animal_tracks.png", dpi=300)
    plt.show()
