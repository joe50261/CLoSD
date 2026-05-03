// Verify the lower-body mask matches the expected feature partition.

import { describe, expect, it } from "vitest";
import {
  HML_FEATURE_DIM,
  HML_LOWER_BODY_MASK,
  HML_UPPER_BODY_MASK,
} from "./humanml.js";

describe("HML_LOWER_BODY_MASK", () => {
  it("has length 263", () => {
    expect(HML_LOWER_BODY_MASK.length).toBe(HML_FEATURE_DIM);
  });

  it("root + foot contact regions are all True", () => {
    // root: indices [0..4)
    for (let i = 0; i < 4; i++) {
      expect(HML_LOWER_BODY_MASK[i]).toBe(true);
    }
    // foot contact: last 4 indices
    for (let i = HML_FEATURE_DIM - 4; i < HML_FEATURE_DIM; i++) {
      expect(HML_LOWER_BODY_MASK[i]).toBe(true);
    }
  });

  it("ric_data of left_knee (joint 4) is True (lower body)", () => {
    // ric_data starts at 4, joint 1 is at offset 0 within ric_data,
    // joint 4 is at offset (4-1)*3 = 9, so global index 4 + 9 = 13.
    expect(HML_LOWER_BODY_MASK[13]).toBe(true);
    expect(HML_LOWER_BODY_MASK[14]).toBe(true);
    expect(HML_LOWER_BODY_MASK[15]).toBe(true);
  });

  it("ric_data of left_wrist (joint 20) is False (upper body)", () => {
    // joint 20 ric offset within ric_data: (20-1)*3 = 57, global = 4 + 57 = 61.
    expect(HML_LOWER_BODY_MASK[61]).toBe(false);
    expect(HML_LOWER_BODY_MASK[62]).toBe(false);
    expect(HML_LOWER_BODY_MASK[63]).toBe(false);
  });

  it("ric_data + rot_data + local_velocity + foot_contact totals to 263", () => {
    // 4 root + 21*3 ric + 21*6 rot + 22*3 vel + 4 foot
    // = 4 + 63 + 126 + 66 + 4 = 263
    expect(4 + 21 * 3 + 21 * 6 + 22 * 3 + 4).toBe(263);
  });

  it("HML_UPPER_BODY_MASK is the complement of LOWER", () => {
    for (let i = 0; i < HML_FEATURE_DIM; i++) {
      expect(HML_UPPER_BODY_MASK[i]).toBe(!HML_LOWER_BODY_MASK[i]);
    }
  });
});
