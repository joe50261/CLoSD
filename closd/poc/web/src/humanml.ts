// HumanML3D feature partition constants.
//
// Mirrors closd/diffusion_planner/data_loaders/humanml_utils.py.
// 263-dim feature layout:
//   [0..4)        root: rot_velocity (1) + linear_velocity (2) + root_y (1)
//   [4..67)       ric_data:        (J-1) * 3   joint xyz   (joints 1..21)
//   [67..193)     rot_data:        (J-1) * 6   joint rot6d (joints 1..21)
//   [193..259)    local_velocity:  J * 3       joint vel   (joints 0..21)
//   [259..263)    foot_contact:    4

export const HML_JOINT_NAMES = [
  "pelvis",
  "left_hip",
  "right_hip",
  "spine1",
  "left_knee",
  "right_knee",
  "spine2",
  "left_ankle",
  "right_ankle",
  "spine3",
  "left_foot",
  "right_foot",
  "neck",
  "left_collar",
  "right_collar",
  "head",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
] as const;

export const NUM_HML_JOINTS = HML_JOINT_NAMES.length; // 22
export const HML_FEATURE_DIM = 263;

const LOWER_BODY_JOINT_NAMES = [
  "pelvis",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_foot",
  "right_foot",
] as const;

const LOWER_BODY_JOINT_BINARY: readonly boolean[] = HML_JOINT_NAMES.map((n) =>
  (LOWER_BODY_JOINT_NAMES as readonly string[]).includes(n),
);

/**
 * Boolean mask of length 263. True at indices that belong to lower-body
 * features. Computed exactly as HML_LOWER_BODY_MASK in humanml_utils.py.
 */
export const HML_LOWER_BODY_MASK: readonly boolean[] = (() => {
  const mask: boolean[] = [];
  // root: 4 entries — all considered "lower body" (pelvis-level info)
  for (let i = 0; i < 4; i++) mask.push(true);
  // ric_data: joints 1..21, x3 each
  for (let j = 1; j < NUM_HML_JOINTS; j++) {
    for (let k = 0; k < 3; k++) mask.push(LOWER_BODY_JOINT_BINARY[j]!);
  }
  // rot_data: joints 1..21, x6 each
  for (let j = 1; j < NUM_HML_JOINTS; j++) {
    for (let k = 0; k < 6; k++) mask.push(LOWER_BODY_JOINT_BINARY[j]!);
  }
  // local_velocity: joints 0..21, x3 each
  for (let j = 0; j < NUM_HML_JOINTS; j++) {
    for (let k = 0; k < 3; k++) mask.push(LOWER_BODY_JOINT_BINARY[j]!);
  }
  // foot contact: 4 entries
  for (let i = 0; i < 4; i++) mask.push(true);
  if (mask.length !== HML_FEATURE_DIM) {
    throw new Error(`HML_LOWER_BODY_MASK length ${mask.length} != ${HML_FEATURE_DIM}`);
  }
  return mask;
})();

export const HML_UPPER_BODY_MASK: readonly boolean[] = HML_LOWER_BODY_MASK.map(
  (b) => !b,
);
